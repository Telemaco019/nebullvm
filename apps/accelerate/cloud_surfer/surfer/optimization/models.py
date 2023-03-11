from dataclasses import dataclass
from typing import Optional, Any, List

from nebullvm.operations.inference_learners.base import BaseInferenceLearner
from nebullvm.tools.base import DeepLearningFramework
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
    model_info: ModelInfo
    latency: float
    throughput: float


@dataclass
class OptimizeInferenceResult:
    """The result of the OptimizeInferenceOp"""

    original_model: OriginalModel
    hardware_setup: HardwareSetup
    optimized_models: List[OptimizedModel]
