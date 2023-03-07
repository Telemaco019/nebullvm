from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import aiofiles
import yaml

import surfer
from surfer.core import constants
from surfer.core.schemas import ExperimentConfig, ExperimentResult, \
    SurferConfig
from surfer.core.util import tmp_dir_clone, copy_files


class ExperimentStatus(str, Enum):
    PENDING = "pending", ":warning:"
    RUNNING = "running", ":play_button:"
    SUCCEEDED = "completed", ":white_check_mark:"
    STOPPED = "stopped", ":stop_button:"
    FAILED = "failed", ":x:"
    UNKNOWN = "unknown", "[?]"

    def __new__(cls, *values):
        obj = super().__new__(cls)
        obj._value_ = values[0]
        obj.icon = values[1]
        return obj

    def __str__(self):
        return "{} {}".format(self.value, self.icon)


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
        experiment_creation_time = datetime.strptime(
            relative_path.parts[1],
            constants.INTERNAL_DATETIME_FORMAT,
        )
        return cls(
            experiment_name=experiment_name,
            experiment_creation_time=experiment_creation_time,
        )

    def as_path(self) -> Path:
        return Path(
            constants.EXPERIMENTS_STORAGE_PREFIX,
            self.experiment_name,
            self.experiment_creation_time.strftime(
                constants.INTERNAL_DATETIME_FORMAT,
            ),
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
    additional_info: Optional[str] = None


@dataclass
class ExperimentDetails:
    summary: ExperimentSummary
    jobs: List[JobSummary]
    result: Optional[ExperimentResult]

    @property
    def name(self):
        return self.summary.name

    @property
    def created_at(self):
        return self.summary.created_at

    @property
    def status(self):
        return self.summary.status


@dataclass
class JobWorkingDir:
    """Working directory of a Ray Job."""
    path: Path
    surfer_config_path: Path
    model_loader_path: Path
    data_loader_path: Path
    model_evaluator_path: Optional[Path] = None

    def __post_init__(self):
        assert self.model_loader_path.is_relative_to(self.path)
        assert self.data_loader_path.is_relative_to(self.path)
        assert self.surfer_config_path.is_relative_to(self.path)
        if self.model_evaluator_path is not None:
            assert self.model_evaluator_path.is_relative_to(self.path)


@asynccontextmanager
async def job_working_dir(
    surfer_config: SurferConfig,
    experiment_config: ExperimentConfig,
) -> JobWorkingDir:
    # Clone config for preventing side effects
    surfer_config = surfer_config.copy()
    async with tmp_dir_clone(Path(surfer.__file__).parent) as tmp:
        # Copy experiment req modules
        modules = [
            experiment_config.model_loader_module,
            experiment_config.data_loader_module,
        ]
        if experiment_config.model_evaluator_module is not None:
            modules.append(experiment_config.model_evaluator_module)
        await copy_files(*modules, dst=tmp)
        # Generate surfer config file
        surfer_config_path = tmp / constants.SURFER_CONFIG_FILE_NAME
        await copy_files(surfer_config.cluster_file, dst=tmp)
        surfer_config.cluster_file = tmp / surfer_config.cluster_file.name
        async with aiofiles.open(surfer_config_path, "w+") as f:
            content = yaml.safe_dump(surfer_config.dict())
            await f.write(content)
        # Create working dir object
        working_dir = JobWorkingDir(
            path=tmp,
            surfer_config_path=surfer_config_path,
            model_loader_path=tmp / experiment_config.model_loader_module.name,
            data_loader_path=tmp / experiment_config.data_loader_module.name,
        )
        if experiment_config.model_evaluator_module is not None:
            working_dir.model_evaluator_path = (
                tmp / experiment_config.model_evaluator_module.name
            )
        yield working_dir
