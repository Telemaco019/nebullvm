from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Any

from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.tools.base import DeepLearningFramework
from surfer.utilities.nebullvm_utils import HardwareSetup


@dataclass
class OptimizedModel:
    inference_learner: BaseInferenceLearner
    latency: float
    metric_drop: float
    technique: str
    compiler: str
    throughput: float
    size_mb: float


@dataclass
class OriginalModel:
    model: Any
    latency: float
    throughput: float
    name: str
    size_mb: float
    framework: DeepLearningFramework


@dataclass
class OptimizeInferenceResult:
    """The result of the OptimizeInferenceOp"""

    original_model: OriginalModel
    hardware_setup: HardwareSetup
    optimized_model: Optional[OptimizedModel]

    @cached_property
    def latency_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.optimized_model.latency == 0:
            return -1
        return self.original_model.latency / self.optimized_model.latency

    @cached_property
    def throughput_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.original_model.throughput == 0:
            return -1
        return self.optimized_model.throughput / self.original_model.throughput

    @cached_property
    def size_improvement_rate(self) -> Optional[float]:
        if self.optimized_model is None:
            return None
        if self.optimized_model.size_mb == 0:
            return -1
        return self.original_model.size_mb / self.optimized_model.size_mb
