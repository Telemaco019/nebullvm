from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Any, List

from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.tools.base import DeepLearningFramework
from surfer.utilities import nebullvm_utils
from surfer.utilities.nebullvm_utils import HardwareSetup


@dataclass
class ModelInfo:
    model_name: str
    model_size_mb: float
    framework: DeepLearningFramework


@dataclass
class OptimizedModel:
    inference_learner: BaseInferenceLearner
    latency: float
    metric_drop: float
    technique: str
    compiler: str
    throughput: float
    size_mb: Optional[float]

    @cached_property
    def model_id(self) -> Optional[str]:
        return nebullvm_utils.generate_model_id(self.inference_learner)


@dataclass
class OriginalModel:
    model: Any
    model_info: ModelInfo
    latency: float
    throughput: float

    @cached_property
    def model_id(self) -> str:
        return nebullvm_utils.generate_model_id(self.model)


@dataclass
class OptimizeInferenceResult:
    """The result of the OptimizeInferenceOp"""

    original_model: OriginalModel
    hardware_setup: HardwareSetup
    optimized_models: List[OptimizedModel]

    @cached_property
    def lowest_latency_model(self) -> Optional[OptimizedModel]:
        # fmt: off
        models = [
            m for m in self.optimized_models
            if m.inference_learner is not None
        ]
        # fmt: on
        if len(models) == 0:
            return None
        return min(models, key=lambda m: m.latency)

    @cached_property
    def inference_learners(self) -> List[BaseInferenceLearner]:
        return [
            m.inference_learner
            for m in self.optimized_models
            if m.inference_learner is not None
        ]
