from dataclasses import dataclass, field
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
    inference_learner: Optional[BaseInferenceLearner]
    latency: float
    metric_drop: float
    technique: str
    compiler: str
    throughput: float
    model_size_mb: Optional[float]
    __id: str = field(default=None, init=False)

    @property
    def model_id(self) -> Optional[str]:
        if self.inference_learner is None:
            return None
        if self.__id is None:
            self.__id = nebullvm_utils.generate_model_id(
                self.inference_learner,
            )
        return self.__id


@dataclass
class OriginalModel:
    model: Any
    model_info: ModelInfo
    latency: float
    throughput: float
    __id: str

    @property
    def mode_id(self) -> str:
        if self.__id is None:
            self.__id = nebullvm_utils.generate_model_id(self.model)
        return self.__id


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
