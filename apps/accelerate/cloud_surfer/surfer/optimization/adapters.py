from typing import List, Optional, Any

from nebullvm.operations.conversions.huggingface import convert_hf_model
from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.operations.optimizations.base import Optimizer
from nebullvm.tools.base import Device
from nebullvm.tools.feedback_collector import FeedbackCollector
from nebullvm.tools.utils import is_huggingface_data
from surfer.optimization.models import OptimizedModel
from surfer.utilities import nebullvm_utils


class HuggingFaceConverter:
    def __init__(self, model: Any, data: List, device: Device):
        self.model = model
        self.data = data
        self.device = device
        self.__converted = False
        self.__hf_model = None
        self.__hf_input_names = None
        self.__hf_output_type = None
        self.__hf_output_structure = None

    def __convert_model(self):
        if not is_huggingface_data(self.data[0]):
            raise ValueError("Cannot convert non-HuggingFace data")
        (
            model,
            data,
            input_names,
            output_structure,
            output_type,
        ) = convert_hf_model(self.model, self.data, self.device)
        self.__hf_model = model
        self.__hf_input_names = input_names
        self.__hf_output_type = output_type
        self.__hf_output_structure = output_structure
        self.__converted = True

    @property
    def hf_model(self):
        if self.__converted is False:
            self.__convert_model()
        return self.__hf_model

    @property
    def hf_input_names(self):
        if self.__converted is False:
            self.__convert_model()
        return self.__hf_input_names

    @property
    def hf_output_structure(self):
        if self.__converted is False:
            self.__convert_model()
        return self.__hf_output_structure

    @property
    def hf_output_type(self):
        if self.__converted is False:
            return self.__hf_output_type


class OptimizerAdapter:
    def __init__(
        self,
        optimizer: Optimizer,
        hf_converter: HuggingFaceConverter,
        batch_size: int,
        input_data: List,
    ):
        self.collector = FeedbackCollector("", "", "")
        self.optimizer = optimizer
        self.optimizer.set_feedback_collector(self.collector)
        self.hf_converter = hf_converter
        self._batch_size = batch_size
        self._input_data = input_data

    def to(self, device: Device) -> "OptimizerAdapter":
        self.optimizer.to(device)
        return self

    def execute(self, *args, **kwargs) -> "OptimizerAdapter":
        self.optimizer.execute(*args, **kwargs)
        return self

    def _wrap_hf_learner(
        self,
        base: Optional[BaseInferenceLearner],
    ) -> Optional[BaseInferenceLearner]:
        if base is None:
            return None

        if is_huggingface_data(self._input_data[0]):
            from nebullvm.operations.inference_learners.huggingface import (
                HuggingFaceInferenceLearner,
            )

            return HuggingFaceInferenceLearner(
                core_inference_learner=base,
                output_structure=self.hf_converter.hf_output_structure,
                input_names=self.hf_converter.hf_input_names,
                output_type=self.hf_converter.hf_output_type,
            )

        return base

    def get_result(self) -> List[OptimizedModel]:
        """
        TODO - This is a temporary solution to merge the results of
        optimization operations. This is brittle -> Nebullvm operation
        must return a class containing all the necessary information

        """
        res = []
        # Merge models returned by th operation with the
        # latencies stored internally in the feedback collector
        for technique_result, optimized_model_tuple in zip(
            self.collector.get("optimizations"), self.optimizer.get_result()
        ):
            metric_drop = optimized_model_tuple[2]
            compiler = technique_result["compiler"]
            technique = technique_result["technique"]
            latency = technique_result["latency"]
            throughput = nebullvm_utils.get_throughput(
                latency,
                self._batch_size,
            )
            inference_learner = self._wrap_hf_learner(optimized_model_tuple[0])
            # Compute model size
            model_size_mb = None
            if inference_learner is not None:
                model_size_mb = inference_learner.get_size()
            # Add to results
            res.append(
                OptimizedModel(
                    inference_learner=inference_learner,
                    metric_drop=metric_drop,
                    compiler=compiler,
                    technique=technique,
                    latency=latency,
                    throughput=throughput,
                    model_size_mb=model_size_mb,
                )
            )
        return res

    def free_model_gpu(self, model):
        self.optimizer.free_model_gpu(model)
