from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Callable, List, Union, Dict, Optional

from nebullvm.config import TRAIN_TEST_SPLIT_RATIO
from nebullvm.operations.base import Operation
from nebullvm.operations.measures.measures import LatencyOriginalModelMeasure
from nebullvm.operations.measures.utils import QUANTIZATION_METRIC_MAP
from nebullvm.operations.optimizations.optimizers import (
    PytorchOptimizer,
    TensorflowOptimizer,
    ONNXOptimizer,
)
from nebullvm.operations.optimizations.utils import \
    map_compilers_and_compressors
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import DataLoader as TorchDataLoader
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
)
from nebullvm.tools.data import DataManager
from nebullvm.tools.utils import (
    is_huggingface_data,
    check_input_data,
    is_data_subscriptable,
    get_dl_framework,
    extract_info_from_data,
)
from surfer.optimization import types
from surfer.optimization.adapters import OptimizerAdapter, HuggingFaceConverter
from surfer.optimization.models import (
    OptimizedModel,
    OriginalModel,
    OptimizeInferenceResult,
)
from surfer.utilities import nebullvm_utils


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
        input_data: types.InputData,
        metric_drop_ths: float = None,
        metric: Union[str, Callable] = None,
        optimization_time: str = "constrained",
        dynamic_info: Dict = None,
        config_file: str = None,
        ignore_compilers: List[str] = None,
        ignore_compressors: List[str] = None,
        store_latencies: bool = False,
        **kwargs,
    ) -> OptimizeInferenceResult:
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
        check_dependencies(self.device)

        ignore_compilers = map_compilers_and_compressors(
            ignore_compilers, ModelCompiler
        )
        ignore_compressors = map_compilers_and_compressors(
            ignore_compressors, ModelCompressor
        )

        optimization_time = OptimizationTime(optimization_time)

        data = input_data

        if isinstance(data, (TorchDataLoader, tf.data.Dataset)):
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

        hf_converter = HuggingFaceConverter(model, data, self.device)
        if is_huggingface_data(data[0]):
            model = hf_converter.hf_model
            data = hf_converter.data
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
        original_model = OriginalModel(
            model=model,
            latency=original_latency,
            name=nebullvm_utils.get_model_name(model),
            size_mb=nebullvm_utils.get_model_size_mb(model),
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
                    hf_converter=hf_converter,
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

        # Check if at least one optimized model has been created
        no_optimized_models = len(optimized_models) < 1
        no_inference_learners = all(
            o.inference_learner is None for o in optimized_models
        )
        if no_optimized_models or no_inference_learners:
            self.logger.warning(
                "No optimized model has been created. This is likely "
                "due to a bug in Speedster. Please open an issue and "
                "report in details your use case."
            )

        # Extract lowest-latency model
        lowest_latency = self._extract_lowest_latency_model(optimized_models)

        return OptimizeInferenceResult(
            original_model=original_model,
            optimized_model=lowest_latency,
            hardware_setup=hw_setup,
        )

    def _optimize(
        self,
        model: Any,
        model_outputs: Iterable,
        hf_converter: HuggingFaceConverter,
        input_data: types.InputData,
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
            hf_converter=hf_converter,
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

    @staticmethod
    def _extract_lowest_latency_model(
        models: List[OptimizedModel],
    ) -> Optional[OptimizedModel]:
        # fmt: off
        inference_learner_models = [
            m for m in models
            if m.inference_learner is not None
        ]
        # fmt: on
        if len(inference_learner_models) == 0:
            return None
        return min(inference_learner_models, key=lambda m: m.latency)

    def get_result(self) -> Any:
        raise NotImplementedError(
            "get_result() is not implemented for this operation",
        )
