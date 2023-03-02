import datetime
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List

from surfer.core import constants
from surfer.core.schemas import ExperimentConfig


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class SubmitExperimentRequest:
    config: ExperimentConfig
    name: str


@dataclass
class ExperimentPath:
    """Path pointing to an experiment data stored on a cloud storage."""

    experiment_name: str
    experiment_creation_time: datetime

    @classmethod
    def from_path(cls, path: Path) -> "ExperimentPath":
        """
        The path is expected to have the following format:
        <EXPERIMENTS_STORAGE_PREFIX>/<experiment_name>/<DATETIME_FORMAT>

        Raises
        ------
            ValueError: If the format of the Path is not valid.
        """
        relative_path = path.relative_to(constants.EXPERIMENTS_STORAGE_PREFIX)
        experiment_name = relative_path.parts[0]
        experiment_creation_time = datetime.strptime(relative_path.parts[1], constants.INTERNAL_DATETIME_FORMAT)
        return cls(
            experiment_name=experiment_name,
            experiment_creation_time=experiment_creation_time,
        )

    def as_path(self) -> Path:
        return Path(
            constants.EXPERIMENTS_STORAGE_PREFIX,
            self.experiment_name,
            self.experiment_creation_time.strftime(constants.INTERNAL_DATETIME_FORMAT),
        )


@dataclass
class ExperimentSummary:
    name: str
    created_at: datetime
    status: ExperimentStatus = ExperimentStatus.UNKNOWN


@dataclass
class JobSummary:
    status: str
    job_id: str


@dataclass
class ExperimentDetails:
    summary: ExperimentSummary
    jobs: List[JobSummary]

    @property
    def name(self):
        return self.summary.name

    @property
    def created_at(self):
        return self.summary.created_at

    @property
    def status(self):
        return self.summary.status
