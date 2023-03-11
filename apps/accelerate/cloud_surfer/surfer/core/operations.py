from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Callable, List, Union, Sequence, Dict, \
    Optional

from pydantic.main import BaseModel

from nebullvm.config import TRAIN_TEST_SPLIT_RATIO
from nebullvm.operations.base import Operation
from nebullvm.operations.conversions.huggingface import convert_hf_model
from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.operations.measures.measures import LatencyOriginalModelMeasure
from nebullvm.operations.measures.utils import QUANTIZATION_METRIC_MAP
from nebullvm.operations.optimizations.base import Optimizer
from nebullvm.operations.optimizations.optimizers import (
    PytorchOptimizer,
    TensorflowOptimizer,
    ONNXOptimizer,
)
from nebullvm.operations.optimizations.utils import \
    map_compilers_and_compressors
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from nebullvm.optional_modules.utils import (
    check_dependencies,
)
from nebullvm.tools.base import (
    OptimizationTime,
    ModelCompiler,
    ModelCompressor,
    DeepLearningFramework,
    DeviceType,
    ModelParams,
    Device,
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.feedback_collector import FeedbackCollector
from nebullvm.tools.utils import (
    is_huggingface_data,
    check_input_data,
    is_data_subscriptable,
    get_dl_framework,
    extract_info_from_data,
)
from surfer import DataLoader
from surfer.utilities import nebullvm_utils
from surfer.utilities.nebullvm_utils import HardwareSetup


@dataclass
class ModelInfo:
    model_name: str
    model_size_mb: float
    framework: DeepLearningFramework


@dataclass
class OptimizedModel:
    inference_learner: Optional[BaseInferenceLearner]
    latency: float
    metric_drop: float
    technique: str
    compiler: str
    throughput: float
    model_size_mb: Optional[float]


@dataclass
class OriginalModel:
    model: Any
    framework: DeepLearningFramework
    model_info: ModelInfo
    latency: float
    throughput: float


class OptimizationResult(BaseModel):
    class Config:
        frozen = True
        extra_fields = "forbid"

    original_model: OriginalModel
    hardware_setup: HardwareSetup
    optimized_models: List[OptimizedModel]


class OptimizeInferenceOp(Operation):
    @staticmethod
    def _as_data_manager(data) -> DataManager:
        if isinstance(data, DataManager):
            return data
        if check_input_data(data) is False:
            raise ValueError(
                "The provided data does not match the expected "
                "format.\n"
                "Speedster supports data in the following formats: \n"
                "- PyTorch DataLoader\n"
                "- TensorFlow Dataset\n"
                "- List of tuples: [((input_0, ... ), label), ...] \n"
                "Inputs and labels should be either tensors or numpy "
                "arrays,\n"
                "depending on the framework used.\n"
            )
        if is_data_subscriptable(data):
            return DataManager(data)
        else:
            return DataManager.from_iterable(data)

    def execute(
        self,
        model: Any,
        input_data: Union[Iterable, Sequence, DataManager],
        metric_drop_ths: float = None,
        metric: Union[str, Callable] = None,
        optimization_time: str = "constrained",
        dynamic_info: Dict = None,
        config_file: str = None,
        ignore_compilers: List[str] = None,
        ignore_compressors: List[str] = None,
        store_latencies: bool = False,
        **kwargs,
    ):
        if model is None:
            raise ValueError("Input model cannot be None")
        if len(input_data) == 0:
            raise ValueError("Input data cannot be empty")

        self.logger.info(
            "running optimization on {}{}".format(
                self.device.type.name,
                self.device.idx if self.device.type is DeviceType.GPU else "",
            )
        )

        hw_setup = nebullvm_utils.get_hw_setup(self.device)
        self.logger.info("hardware setup", hw_setup.json(indent=2))

        check_dependencies(self.device)

        ignore_compilers = map_compilers_and_compressors(
            ignore_compilers, ModelCompiler
        )
        ignore_compressors = map_compilers_and_compressors(
            ignore_compressors, ModelCompressor
        )

        optimization_time = OptimizationTime(optimization_time)

        data = input_data

        if isinstance(data, (DataLoader, tf.data.Dataset)):
            try:
                data = DataManager.from_dataloader(data)
            except Exception:
                raise ValueError(
                    "The provided dataloader does not match the expected "
                    "format.\n"
                    "Speedster supports dataloaders that return tuples in "
                    "the\n"
                    "following formats: \n"
                    "Single input: (input,  label)\n"
                    "Multiple inputs: ((input1, input2, ...),  label) or "
                    "(input1, input2, ...,  label)\n"
                    "Inputs and labels should be either tensors or numpy "
                    "arrays,\n"
                    "depending on the framework used.\n"
                )

        if is_huggingface_data(data[0]):
            (
                model,
                data,
                input_names,
                output_structure,
                output_type,
            ) = convert_hf_model(model, data, self.device, **kwargs)

            if dynamic_info is None:
                self.logger.warning(
                    "Dynamic shape info has not been provided for the "
                    "HuggingFace model. The resulting optimized model "
                    "will be usable only with a fixed input shape. "
                    "To optimize the model for dynamic shapes, please "
                    "look here: https://nebuly.gitbook.io/nebuly/modules/"
                    "speedster/how-to-guides"
                    "#using-dynamic-shape."
                )

        data = self._as_data_manager(data)
        dl_framework = get_dl_framework(model)

        if metric_drop_ths is not None and metric_drop_ths <= 0:
            metric_drop_ths = None
        elif metric_drop_ths is not None and metric is None:
            metric = "numeric_precision"
        if isinstance(metric, str):
            metric = QUANTIZATION_METRIC_MAP.get(metric)

        model_params: ModelParams = extract_info_from_data(
            model=model,
            input_data=data,
            dl_framework=dl_framework,
            dynamic_info=dynamic_info,
            device=self.device,
        )

        data.split(TRAIN_TEST_SPLIT_RATIO)

        # -------- Benchmark original model --------
        original_latency_op = LatencyOriginalModelMeasure().to(self.device)
        original_latency_op.execute(
            model=model,
            input_data=data.get_split("test"),
            dl_framework=dl_framework,
        )
        original_latency = original_latency_op.get_result()[1]
        model_info = ModelInfo(
            model_name=nebullvm_utils.get_model_name(model),
            model_size_mb=nebullvm_utils.get_model_size_mb(model),
            framework=dl_framework,
        )
        original_model = OriginalModel(
            model=model,
            model_info=model_info,
            latency=original_latency,
            framework=dl_framework,
            throughput=nebullvm_utils.get_throughput(
                latency=original_latency,
                batch_size=model_params.batch_size,
            ),
        )
        # ------------------------------------------

        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir) / "fp32"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Convert model to all available frameworks
            conversion_op = nebullvm_utils.get_conversion_op(dl_framework)
            conversion_op.to(self.device).set_state(model, data).execute(
                save_path=tmp_dir,
                model_params=model_params,
            )

            # Optimize models
            optimized_models: List[OptimizedModel] = []
            for model in conversion_op.get_result():
                optimized_models += self._optimize(
                    model=model,
                    input_data=data,
                    model_outputs=original_latency_op.get_result()[0],
                    optimization_time=optimization_time,
                    metric_drop_ths=metric_drop_ths,
                    metric=metric,
                    model_params=model_params,
                    ignore_compilers=ignore_compilers,
                    ignore_compressors=ignore_compressors,
                    source_dl_framework=dl_framework,
                )

        optimized_models.sort(key=lambda x: x.latency, reverse=False)

        if len(optimized_models) < 1 or optimized_models[0].inference_learner is None:
            raise RuntimeError(
                "No optimized model has been created. This is likely "
                "due to a bug in Speedster. Please open an issue and "
                "report in details your use case."
            )

        if is_huggingface_data(data[0]):
            from nebullvm.operations.inference_learners.huggingface import (
                HuggingFaceInferenceLearner,
            )

            optimal_inference_learner = HuggingFaceInferenceLearner(
                core_inference_learner=optimized_models[0].inference_learner,
                output_structure=output_structure,
                input_names=input_names,
                output_type=output_type,
            )
        else:
            optimal_inference_learner = optimized_models[0].inference_learner

        return OptimizationResult(
            original_model=original_model,
            optimized_models=optimized_models,
            hardware_setup=hw_setup,
        )

    def _optimize(
        self,
        model: Any,
        model_outputs: Iterable,
        input_data: Iterable,
        optimization_time: OptimizationTime,
        metric_drop_ths: float,
        metric: Callable,
        model_params: ModelParams,
        ignore_compilers: List[ModelCompiler],
        ignore_compressors: List[ModelCompressor],
        source_dl_framework: DeepLearningFramework,
    ) -> List[OptimizedModel]:
        if isinstance(model, torch.nn.Module):
            optimization_op = PytorchOptimizer()
        elif isinstance(model, tf.Module):
            optimization_op = TensorflowOptimizer()
        else:
            optimization_op = ONNXOptimizer()

        # Add adapter for output results
        optimization_op = OptimizerAdapter(
            optimizer=optimization_op,
            batch_size=model_params.batch_size,
            input_data=input_data,
        )

        # Run optimization
        optimized_models = (
            optimization_op.to(self.device)
            .execute(
                model=model,
                input_data=input_data,
                optimization_time=optimization_time,
                metric_drop_ths=metric_drop_ths,
                metric=metric,
                model_params=model_params,
                model_outputs=model_outputs,
                ignore_compilers=ignore_compilers,
                ignore_compressors=ignore_compressors,
                source_dl_framework=source_dl_framework,
            )
            .get_result()
        )

        if isinstance(model, torch.nn.Module):
            optimization_op.free_model_gpu(model)

        return optimized_models

    def get_result(self) -> Any:
        raise NotImplementedError(
            "get_result() is not implemented for this operation",
        )


class OptimizerAdapter:
    def __init__(
        self,
        optimizer: Optimizer,
        batch_size: int,
        input_data: Iterable,
    ):
        self.collector = FeedbackCollector("", "", "")
        self.optimizer = optimizer
        self.optimizer.set_feedback_collector(self.collector)
        self._batch_size = batch_size
        self._input_data = input_data

    def to(self, device: Device) -> "OptimizerAdapter":
        self.optimizer.to(device)
        return self

    def execute(self, *args, **kwargs) -> "OptimizerAdapter":
        self.optimizer.execute(*args, **kwargs)
        return self

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
            inference_learner: Optional[BaseInferenceLearner]
            inference_learner = optimized_model_tuple[0]
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
