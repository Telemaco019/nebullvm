import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List

from pydantic.dataclasses import dataclass
from pydantic.main import BaseModel


class ExperimentConfig(BaseModel):
    description: Optional[str]
    data_loader_module: Path
    model_loader_module: Path
    model_evaluator_module: Optional[Path]
    additional_requirements: List[str] = []

    class Config:
        extra = "forbid"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubmitExperimentRequest:
    config: ExperimentConfig
    name: str


@dataclass
class ExperimentSummary:
    name: str
    status: ExperimentStatus
    created_at: datetime.datetime
