import datetime
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List

from pydantic.dataclasses import dataclass
from pydantic.main import BaseModel

from surfer.core import constants


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
    UNKNOWN = "unknown"


@dataclass
class SubmitExperimentRequest:
    config: ExperimentConfig
    name: str


@dataclass
class ExperimentSummary:
    name: str
    created_at: datetime
    status: ExperimentStatus = ExperimentStatus.UNKNOWN

    @classmethod
    def from_path(cls, path: Path) -> "ExperimentSummary":
        """Create an ExperimentSummary from a path pointing to the experiment data.

        The path is expected to have the following format:
        <EXPERIMENTS_STORAGE_PREFIX>/<DATETIME_FORMAT>/<experiment_name>

        Raises
        ------
            ValueError: If the format of the Path is not valid.
        """
        relative_path = path.relative_to(constants.EXPERIMENTS_STORAGE_PREFIX)
        created_at = datetime.strptime(relative_path.parts[0], constants.DATETIME_FORMAT)
        name = relative_path.parts[1]
        return cls(name=name, created_at=created_at)  # type: ignore


@dataclass
class ExperimentDetails:
    pass
