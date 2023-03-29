import abc
import copy
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Any, Union

from nebullvm.operations.conversions.huggingface import convert_hf_model
from nebullvm.operations.inference_learners.base import (
    BaseInferenceLearner,
    LearnerMetadata,
)
from nebullvm.operations.optimizations.base import Optimizer
from nebullvm.optional_modules.diffusers import StableDiffusionPipeline
from nebullvm.optional_modules.torch import torch
from nebullvm.tools.base import Device, DeviceType
from nebullvm.tools.diffusers import (
    get_unet_inputs,
    preprocess_diffusers,
    postprocess_diffusers,
)
from nebullvm.tools.feedback_collector import FeedbackCollector
from nebullvm.tools.utils import is_huggingface_data
from surfer.optimization import types
from surfer.optimization.models import OptimizedModel
from surfer.utilities import nebullvm_utils


class ModelAdapter(abc.ABC):
    @property
    @abstractmethod
    def adapted_model(self):
        pass

    @property
    @abstractmethod
    def adapted_data(self):
        pass

    @abstractmethod
    def adapt_inference_learner(self, model) -> BaseInferenceLearner:
        pass


# TODO: move to Nebullvm
class DiffusionInferenceLearner(BaseInferenceLearner):
    name = "StableDiffusion"

    def __init__(self, pipeline: StableDiffusionPipeline):
        self.pipeline = pipeline

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    def tensor2list(self, tensor: Any) -> List:
        raise NotImplementedError()

    def _read_file(self, input_file: str) -> Any:
        raise NotImplementedError()

    def _save_file(self, prediction: Any, output_file: str):
        raise NotImplementedError()

    def run(self, *args, **kwargs) -> Any:
        self.pipeline(*args, **kwargs)

    def save(self, path: Union[str, Path], **kwargs):
        self.pipeline.unet.model.save(path)

    @classmethod
    def load(
        cls,
        path: Union[Path, str],
        **kwargs,
    ):
        try:
            pipe = kwargs["pipe"]
        except KeyError:
            raise TypeError("Missing required argument 'pipe'")
        optimized_model = LearnerMetadata.read(path).load_model(path)
        return postprocess_diffusers(
            optimized_model,
            pipe,
            optimized_model.device,
        )

    def get_size(self):
        return 0  # TODO - How do we get the size of a diffusion model?

    def free_gpu_memory(self):
        raise NotImplementedError()

    def get_inputs_example(self):
        raise NotImplementedError()

    @property
    def output_format(self):
        return ".pt"

    @property
    def input_format(self):
        return ".pt"

    def list2tensor(self, listified_tensor: List) -> Any:
        raise NotImplementedError()


# TODO: move to Nebullvm
class DiffusionAdapter(ModelAdapter):
    def __init__(
        self,
        original_pipeline: StableDiffusionPipeline,
        data: List,
        device: Device,
    ):
        self.original_pipeline = copy.deepcopy(original_pipeline)
        self.original_data = data
        self.device = device
        self.__adapted = False
        self.__df_model = None
        self.__df_data = None

    def __adapt(self):
        model = copy.deepcopy(self.original_pipeline)
        model.get_unet_inputs = get_unet_inputs
        model.to(self.device.to_torch_format())
        self.__df_data = [
            (
                tuple(
                    d.reshape((1,)) if d.shape == torch.Size([]) else d
                    for d in model.get_unet_inputs(
                        model,
                        prompt=prompt,
                    )
                    if d is not None
                ),
                None,
            )
            for prompt in self.original_data
        ]
        self.__df_model = preprocess_diffusers(model)
        self.__adapted = True

    @property
    def adapted_model(self):
        if self.__adapted is False:
            self.__adapt()
        return self.__df_model

    @property
    def adapted_data(self):
        if self.__adapted is False:
            self.__adapt()
        return self.__df_data

    def adapt_inference_learner(self, model) -> BaseInferenceLearner:
        pipe = copy.deepcopy(self.original_pipeline)
        if self.device.type is DeviceType.GPU:
            pipe.to("cuda")
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        pipe = postprocess_diffusers(
            model,
            pipe,
            self.device,
        )
        return DiffusionInferenceLearner(pipe)


# TODO: move to Nebullvm
class HuggingFaceAdapter(ModelAdapter):
    def __init__(self, model: Any, data: List, device: Device):
        self.original_model = model
        self.original_data = data
        self.device = device
        self.__adapted = False
        self.__hf_model = None
        self.__hf_data = None
        self.__hf_input_names = None
        self.__hf_output_type = None
        self.__hf_output_structure = None

    def __adapt_model(self):
        if not is_huggingface_data(self.original_data[0]):
            raise ValueError("Cannot convert non-HuggingFace data")
        (
            model,
            data,
            input_names,
            output_structure,
            output_type,
        ) = convert_hf_model(
            self.original_model,
            self.original_data,
            self.device,
        )
        self.__hf_model = model
        self.__hf_data = data
        self.__hf_input_names = input_names
        self.__hf_output_type = output_type
        self.__hf_output_structure = output_structure
        self.__adapted = True

    @property
    def adapted_model(self):
        if self.__adapted is False:
            self.__adapt_model()
        return self.__hf_model

    @property
    def adapted_data(self):
        if self.__adapted is False:
            self.__adapt_model()
        return self.__hf_data

    def adapt_inference_learner(self, optimized_model) -> BaseInferenceLearner:
        from nebullvm.operations.inference_learners.huggingface import (
            HuggingFaceInferenceLearner,
        )

        return HuggingFaceInferenceLearner(
            core_inference_learner=optimized_model,
            output_structure=self.__hf_output_structure,
            input_names=self.__hf_input_names,
            output_type=self.__hf_output_type,
        )


class OptimizerAdapter:
    def __init__(
        self,
        optimizer: Optimizer,
        batch_size: int,
        input_data: types.InputData,
        model_adapter: Optional[ModelAdapter] = None,
    ):
        self.collector = FeedbackCollector("", "", "")
        self.optimizer = optimizer
        self.optimizer.set_feedback_collector(self.collector)
        self.model_adapter = model_adapter
        self._batch_size = batch_size
        self._input_data = input_data

    def to(self, device: Device) -> "OptimizerAdapter":
        self.optimizer.to(device)
        return self

    def execute(self, *args, **kwargs) -> "OptimizerAdapter":
        self.optimizer.execute(*args, **kwargs)
        return self

    def _adapt_learner(
        self,
        base: Optional[BaseInferenceLearner],
    ) -> Optional[BaseInferenceLearner]:
        if base is None:
            return None
        if self.model_adapter is None:
            return base
        return self.model_adapter.adapt_inference_learner(base)

    def get_result(self) -> List[OptimizedModel]:
        """
        TODO - This is a temporary solution to merge the results of
        optimization operations. This is brittle -> Nebullvm operation
        must return a class containing all the necessary information

        """
        res = []
        # Merge models returned by th operation with the
        # latencies stored internally in the feedback collector
        #
        # technique_result:
        #   - "compiler" -> str
        #   - "technique" -> str
        #   - "latency" -> float
        #
        # optimized_model_tuple:
        #   [0] -> Inference learner (BaseInferenceLearner)
        #   [1] -> Latency (str)
        #   [2] -> Metric drop (float)
        # fmt: off
        optimizations = [
            o for o in self.collector.get("optimizations", [])
            if o[1] > 0  # Filter out failed optimizations
        ]
        # fmt: on
        for technique_result, optimized_model_tuple in zip(
            optimizations, self.optimizer.get_result()
        ):
            metric_drop = optimized_model_tuple[2]
            compiler = technique_result["compiler"]
            technique = technique_result["technique"]
            latency = technique_result["latency"]
            throughput = nebullvm_utils.get_throughput(
                latency,
                self._batch_size,
            )
            inference_learner = self._adapt_learner(optimized_model_tuple[0])
            # Compute model size
            model_size_mb = None
            if inference_learner is not None:
                model_size_mb = inference_learner.get_size() / 1e6
            # Add to results
            res.append(
                OptimizedModel(
                    inference_learner=inference_learner,
                    metric_drop=metric_drop,
                    compiler=compiler,
                    technique=technique,
                    latency_seconds=latency,
                    throughput=throughput,
                    size_mb=model_size_mb,
                )
            )
        return res

    def free_model_gpu(self, model):
        self.optimizer.free_model_gpu(model)
