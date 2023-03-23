import asyncio
import datetime
import functools
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional

import aiofiles
from pydantic.error_wrappers import ValidationError
from ray.dashboard.modules.job.common import JobStatus
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray.dashboard.modules.job.sdk import JobSubmissionClient

import surfer
from surfer.cli.runner import RunCommandBuilder
from surfer.common import constants, schemas
from surfer.common.exceptions import InternalError, NotFoundError
from surfer.common.schemas import SurferConfig, ExperimentResult
from surfer.log import logger
from surfer.storage import StorageProvider
from surfer.storage.clients import StorageClient
from surfer.storage.models import StorageConfig
from surfer.utilities.file_utils import tmp_dir_clone, copy_files


class ExperimentStatus(str, Enum):
    PENDING = "pending", ":yellow_circle:"
    RUNNING = "running", ":green_circle:"
    SUCCEEDED = "completed", ":white_check_mark:"
    STOPPED = "stopped", ":stop_button:"
    FAILED = "failed", ":cross_mark:"
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
    config: schemas.ExperimentConfig
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
        experiment_creation_time = datetime.datetime.strptime(
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

    def marker_path(self) -> Path:
        """
        Return a path to a marker file used for identifying experiments
        on the cloud storage.
        """
        return self.as_path() / constants.EXPERIMENT_MARKER_FILE_NAME


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
    result: Optional[schemas.ExperimentResult]

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

    base: Path
    surfer_config_path: Path
    model_loader_path: Path
    data_loader_path: Path
    model_evaluator_path: Optional[Path] = None


@asynccontextmanager
async def job_working_dir(
    surfer_config: schemas.SurferConfig,
    experiment_config: schemas.ExperimentConfig,
) -> JobWorkingDir:
    # Clone config for preventing side effects
    surfer_config = surfer_config.copy()
    async with tmp_dir_clone(Path(surfer.__file__).parent) as tmp:
        # Copy experiment req modules
        modules = [
            experiment_config.model_loader,
            experiment_config.data_loader,
        ]
        if experiment_config.model_evaluator is not None:
            modules.append(experiment_config.model_evaluator)
        await copy_files(*modules, dst=tmp)
        # Copy surfer cluster file
        await copy_files(surfer_config.cluster_file, dst=tmp)
        # Generate surfer config file
        surfer_config_path = tmp / constants.SURFER_CONFIG_FILE_NAME
        surfer_config.cluster_file = surfer_config.cluster_file.name
        async with aiofiles.open(surfer_config_path, "w+") as f:
            await f.write(surfer_config.json(indent=2))
        # Create working dir object
        working_dir = JobWorkingDir(
            base=tmp,
            surfer_config_path=Path(surfer_config_path.name),
            model_loader_path=Path(experiment_config.model_loader.name),
            data_loader_path=Path(experiment_config.data_loader.name),
        )
        if experiment_config.model_evaluator is not None:
            working_dir.model_evaluator_path = (
                experiment_config.model_evaluator.name
            )  # noqa E501
        yield working_dir


class ExperimentService:
    def __init__(
        self,
        storage_client: StorageClient,
        job_client: JobSubmissionClient,
        surfer_config: SurferConfig,
    ):
        self.storage_client = storage_client
        self.job_client = job_client
        self.surfer_config = surfer_config

    @staticmethod
    def __can_stop_experiment(status: ExperimentStatus) -> bool:
        return status.value in [
            ExperimentStatus.RUNNING.value,
            ExperimentStatus.PENDING.value,
        ]

    @staticmethod
    def __get_run_cmd(
        experiment_path: ExperimentPath,
        workdir: JobWorkingDir,
        config: schemas.ExperimentConfig,
    ) -> str:
        builder = (
            RunCommandBuilder()
            .with_results_dir(experiment_path.as_path())
            .with_model_loader(workdir.model_loader_path)
            .with_data_loader(workdir.data_loader_path)
            .with_surfer_config(workdir.surfer_config_path)
            .with_ignored_accelerators(config.ignored_accelerators)
            .with_ignored_compilers(config.ignored_compilers)
        )
        if logger.level == logging.DEBUG:
            builder.with_debug()

        return builder.get_command()

    @staticmethod
    def _filter_experiment_jobs(
        jobs: List[JobDetails],
        experiment_name: str,
    ) -> List[JobDetails]:
        return [
            j
            for j in jobs
            if experiment_name
            == j.metadata.get(constants.JOB_METADATA_EXPERIMENT_NAME, None)
        ]

    @staticmethod
    def _get_experiment_status(jobs: List[JobDetails]) -> ExperimentStatus:
        if len(jobs) == 0:
            return ExperimentStatus.UNKNOWN
        # If any job is running, the experiment is running
        if any([j.status is JobStatus.RUNNING for j in jobs]):
            return ExperimentStatus.RUNNING
        # If any job is pending, the experiment is pending
        if any([j.status is JobStatus.PENDING for j in jobs]):
            return ExperimentStatus.PENDING
        # If all jobs are succeeded, the experiment is succeeded
        if all([j.status is JobStatus.SUCCEEDED for j in jobs]):
            return ExperimentStatus.SUCCEEDED
        # If any job is failed, the experiment is failed
        if any([j.status is JobStatus.FAILED for j in jobs]):
            return ExperimentStatus.FAILED
        # If all jobs are stopped, the experiment is stopped
        if all([j.status is JobStatus.STOPPED for j in jobs]):
            return ExperimentStatus.STOPPED
        # Default - the experiment status is unknown
        return ExperimentStatus.UNKNOWN

    async def _fetch_all_jobs(self) -> List[JobDetails]:
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.job_client.list_jobs,
        )

    async def _stop_job(self, job_id: str):
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.job_client.stop_job,
            job_id,
        )

    async def _get_experiment_jobs(
        self,
        experiment_name: str,
    ) -> List[JobDetails]:
        jobs = await self._fetch_all_jobs()
        return self._filter_experiment_jobs(jobs, experiment_name)

    async def _get_experiment_paths(
        self,
        experiment_name: Optional[str] = None,
    ) -> List[ExperimentPath]:
        prefix = f"{constants.EXPERIMENTS_STORAGE_PREFIX}/"
        if experiment_name is not None:
            prefix += f"{experiment_name}/"
        experiment_paths = []
        logger.debug("listing paths with prefix: ", prefix)
        for path in await self.storage_client.list(prefix):
            try:
                p = ExperimentPath.from_path(path)
                if p not in experiment_paths:
                    experiment_paths.append(p)
            except ValueError:
                pass
            except IndexError:
                pass
        return experiment_paths

    async def _fetch_result(
        self,
        experiment_path: ExperimentPath,
    ) -> Optional[ExperimentResult]:
        try:
            path = Path(
                experiment_path.as_path(),
                constants.EXPERIMENT_RESULT_FILE_NAME,
            )
            raw_data = await self.storage_client.get(path)
            if raw_data is None:
                return None
            return ExperimentResult.parse_raw(raw_data)
        except ValidationError as e:
            raise InternalError(f"failed to parse experiment result: {e}")

    async def submit(self, req: SubmitExperimentRequest):
        """Submit a new experiment.

        Submit a new experiment for execution.

        Parameters
        ----------
        req : SubmitExperimentRequest
            The configuration of the experiment to submit.

        Raises
        ------
        InternalError
            If any internal error occurs during experiment submission.
        ValueError
            If an experiment with the same name already exists.
        """
        if await self.get(req.name) is not None:
            raise ValueError(f"experiment {req.name} already exists")
        # Create experiment entry in cloud storage for tracking it
        experiment_path = ExperimentPath(
            experiment_name=req.name,
            experiment_creation_time=datetime.datetime.now(),
        )
        await self.storage_client.upload_content(
            "",
            experiment_path.marker_path(),
        )
        # Submit Ray job
        async with job_working_dir(self.surfer_config, req.config) as workdir:
            # Build run command
            entrypoint = self.__get_run_cmd(
                experiment_path,
                workdir,
                req.config,
            )
            # Build dependencies
            requirements = _get_base_job_requirements(
                self.surfer_config.storage,
            )
            requirements += req.config.additional_requirements
            # Submit Job
            logger.debug(
                "submitting Ray job",
                {
                    "entrypoint": entrypoint,
                    "working_dir": workdir.base.as_posix(),
                    "pip": requirements,
                },
            )
            job_id = self.job_client.submit_job(
                entrypoint=entrypoint,
                runtime_env={
                    "working_dir": workdir.base.as_posix(),
                    "pip": requirements,
                },
                metadata={
                    constants.JOB_METADATA_EXPERIMENT_NAME: req.name,
                },
            )
            logger.debug("job ID", job_id)

    async def delete(self, experiment_name: str):
        """Delete the experiment and all its data

        Delete the experiment with the specified name and remove all its data:
        * metrics
        * optimized models
        * jobs on the Ray cluster

        The experiment can only be deleted if all its jobs are in
        a terminal state (succeeded or failed).

        Parameters
        ----------
        experiment_name: str
            Name of the experiment to delete

        Raises
        ------
        NotFoundError
            If the experiment does not exist
        InternalError
            If the experiment is not in a terminal state,
            or any error occurs during deletion
        """
        # Check if all jobs are in a terminal state
        experiment_jobs = await self._get_experiment_jobs(experiment_name)
        for j in experiment_jobs:
            if j.status is JobStatus.RUNNING:
                raise InternalError(
                    "job {} is still running, stop experiment first".format(
                        j.job_id,
                    )
                )
            if j.status is JobStatus.PENDING:
                raise InternalError(
                    "job {} is still pending, stop experiment first".format(j.job_id)
                )
        # Delete experiment data
        experiment_paths = await self._get_experiment_paths(experiment_name)
        if len(experiment_paths) == 0:
            raise NotFoundError(
                "experiment {} does not exist".format(experiment_name),
            )
        logger.info("deleting experiment data...")
        delete_data_coros = []
        for path in experiment_paths:
            delete_data_coros.append(self.storage_client.delete(path.as_path()))
        try:
            await asyncio.gather(*delete_data_coros)
        except FileNotFoundError as e:
            logger.debug("trying to delete non-existing file: ", e)
        # Delete Jobs
        logger.info("deleting experiment jobs...")
        delete_job_coros = []
        for j in experiment_jobs:
            coro = asyncio.get_event_loop().run_in_executor(
                None, self.job_client.delete_job, j.submission_id
            )
            delete_job_coros.append(coro)
        await asyncio.gather(*delete_job_coros)

    async def stop(self, experiment_name: str):
        """Stop the experiment and all its Jobs

        Stop the experiment with the specified name and all its Jobs
        on the Ray cluster.

        The experiment can only be stopped if it is in a state that
        can be stopped (running or pending).

        Stopping an experiment will not delete any data.
        After stopping, the experiment cannot be resumed.

        Parameters
        ----------
        experiment_name: str
            Name of the experiment to stop

        Raises
        -------
        NotFoundError
            If the experiment does not exist
        ValueError
            If the experiment is not in a state that can be stopped
        InternalError
            If any error occurs during stopping
        """
        # Check if the experiment exists
        paths = await self._get_experiment_paths(experiment_name)
        if len(paths) == 0:
            raise NotFoundError("experiment {} does not exist".format(experiment_name))
        # Check if experiment can be stopped
        experiment_jobs = await self._get_experiment_jobs(experiment_name)
        if len(experiment_jobs) == 0:
            raise ValueError("no jobs found for experiment {}".format(experiment_name))
        status = self._get_experiment_status(experiment_jobs)
        if self.__can_stop_experiment(status) is False:
            raise ValueError("cannot stop {} experiment".format(status.value))
        # Stop experiment jobs
        logger.info("stopping experiment jobs...")
        coros = []
        for j in experiment_jobs:
            coros.append(self._stop_job(j.job_id))
        try:
            await asyncio.gather(*coros)
        except RuntimeError as e:
            raise InternalError("error stopping experiment jobs: {}".format(e))

    async def list(self) -> List[ExperimentSummary]:
        """List all experiments

        List all the current and past experiments. For each experiment,
        only a summary including its essential information is returned.

        Returns
        -------
        List[ExperimentSummary]
            Summary containing essential information of
            each available experiment
        """
        paths = await self._get_experiment_paths()
        # No experiment data is found, we are done
        if len(paths) == 0:
            return []
        # Init summaries
        summaries = [
            ExperimentSummary(
                name=p.experiment_name,
                created_at=p.experiment_creation_time,
            )
            for p in paths
        ]
        # Fetch jobs and update summaries status
        jobs = await self._fetch_all_jobs()
        for summary in summaries:
            experiment_jobs = self._filter_experiment_jobs(jobs, summary.name)
            summary.status = self._get_experiment_status(experiment_jobs)
        return summaries

    async def get(self, experiment_name: str) -> Optional[ExperimentDetails]:
        """Get the details of the specified experiment

        Get all the information of the specified experiment.

        Parameters
        ----------
        experiment_name: str
            Name of the experiment to fetch

        Raises
        ------
        InternalError
            If the experiment results are found but failed to parse

        Returns
        -------
        Optional[ExperimentDetails]
            Experiment details if found, None otherwise
        """
        # Get experiment path
        logger.debug("fetching experiment paths on storage")
        paths = await self._get_experiment_paths(experiment_name)
        if len(paths) == 0:
            logger.debug("no experiment data found")
            return None
        experiment_path = paths[0]
        # Fetch jobs and update summary status
        logger.debug("fetching experiment Ray jobs")
        summary = ExperimentSummary(
            name=experiment_path.experiment_name,
            created_at=experiment_path.experiment_creation_time,
        )
        experiment_jobs = await self._get_experiment_jobs(experiment_name)
        summary.status = self._get_experiment_status(experiment_jobs)
        # Fetch experiment result
        result = None
        if summary.status in [
            ExperimentStatus.SUCCEEDED,
            ExperimentStatus.UNKNOWN,
        ]:
            result = await self._fetch_result(experiment_path)
            if result is None:
                logger.warn(
                    f"experiment {experiment_name} is succeeded, "
                    "but results are missing"
                )
        # Init Experiment details
        job_summaries = []
        for job in experiment_jobs:
            job_summary = JobSummary(
                status=job.status,
                job_id=job.submission_id,
                additional_info=job.message,
            )
            job_summaries.append(job_summary)
        return ExperimentDetails(
            summary=summary,
            jobs=job_summaries,
            result=result,
        )


def new_experiment_service(config: SurferConfig) -> ExperimentService:
    storage_client = StorageClient.from_config(config.storage)
    job_client = JobSubmissionClient(address=config.ray_address)
    return ExperimentService(
        storage_client=storage_client,
        job_client=job_client,
        surfer_config=config,
    )


@functools.cache
def _get_base_job_requirements(storage_config: StorageConfig) -> List[str]:
    def __with_version(req: List[str]) -> List[str]:
        for r in req:
            if len(r.split("==")) == 1:
                yield f"{r}=={version(r)}"
            else:
                yield r

    dependencies = [
        "typer",
        "nebullvm",
        "aiofiles",
        "protobuf==3.20.*",
    ]

    # Add GCP storage dependency
    if storage_config.provider is StorageProvider.GCP:
        dependencies += ["google-cloud-storage"]

    # Add Azure storage dependency
    if storage_config.provider is StorageProvider.AZURE:
        dependencies += [
            "azure-storage-blob",
        ]

    # Add AWS storage dependency
    if storage_config.provider is StorageProvider.AWS:
        dependencies += []  # todo

    requirements = list(__with_version(dependencies))
    return requirements
