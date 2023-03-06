import abc
import asyncio
import json
from pathlib import Path
from typing import List, Optional, Iterable, Dict

import yaml
from pydantic.error_wrappers import ValidationError
from ray.job_submission import JobDetails, JobStatus
from ray.job_submission import JobSubmissionClient

from surfer.core import constants
from surfer.core.exceptions import InternalError, NotFoundError
from surfer.core.models import (
    SubmitExperimentRequest,
    ExperimentSummary,
    ExperimentDetails,
    ExperimentStatus,
    ExperimentPath,
    JobSummary,
)
from surfer.core.schemas import SurferConfig, ExperimentResult
from surfer.log import logger
from surfer.storage.clients import StorageClient


class ModelLoader(abc.ABC):
    @abc.abstractmethod
    def load_model(self, *args, **kwargs) -> any:
        pass


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load_data(self, *args, **kwargs) -> Iterable:
        pass


class ModelEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate_model(self, model, *args, **kwargs) -> Dict[str, any]:
        pass


class DefaultModelEvaluator(ModelEvaluator):
    def evaluate_model(self, model, *args, **kwargs) -> Dict[str, any]:
        pass


class ExperimentService:
    def __init__(
        self,
        storage_client: StorageClient,
        job_client: JobSubmissionClient,
    ):
        self.storage_client = storage_client
        self.job_client = job_client

    @staticmethod
    def __can_stop_experiment(status: ExperimentStatus) -> bool:
        return status.value in [
            ExperimentStatus.RUNNING.value,
            ExperimentStatus.PENDING.value,
        ]

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
        experiment_paths = []
        prefix = f"{constants.EXPERIMENTS_STORAGE_PREFIX}/"
        if experiment_name is not None:
            prefix += f"{experiment_name}/"
        paths = await self.storage_client.list(prefix)
        for path in paths:
            try:
                experiment_paths.append(ExperimentPath.from_path(path))
            except ValueError:
                pass
            except IndexError:
                pass
        return experiment_paths

    async def _fetch_result(
        self,
        path: ExperimentPath,
    ) -> Optional[ExperimentResult]:
        try:
            raw_data = await self.storage_client.get(path.as_path())
            if raw_data is None:
                return None
            return ExperimentResult.parse_raw(raw_data)
        except ValidationError as e:
            raise InternalError(f"failed to parse experiment result: {e}")

    async def submit(self, req: SubmitExperimentRequest):

        self.job_client.submit_job(
            entrypoint="python3 . --help",
            runtime_env={
                "working_dir": "surfer/runner_2",
            },
            metadata={
                constants.JOB_METADATA_EXPERIMENT_NAME: req.name,
            },
        )

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
                    "job {} is still pending, stop experiment first".format(
                        j.job_id
                    )
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
            delete_data_coros.append(
                self.storage_client.delete(path.as_path())
            )
        try:
            await asyncio.gather(*delete_data_coros)
        except FileNotFoundError as e:
            logger.debug("trying to delete non-existing file: ", e)
        # Delete Jobs
        logger.info("deleting experiment jobs...")
        delete_job_coros = []
        for j in experiment_jobs:
            coro = asyncio.get_event_loop().run_in_executor(
                None, self.job_client.delete_job, j.job_id
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
            raise NotFoundError(
                "experiment {} does not exist".format(experiment_name)
            )
        # Check if experiment can be stopped
        experiment_jobs = await self._get_experiment_jobs(experiment_name)
        if len(experiment_jobs) == 0:
            raise ValueError(
                "no jobs found for experiment {}".format(experiment_name)
            )
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
        paths = await self._get_experiment_paths(experiment_name)
        if len(paths) == 0:
            return None
        experiment_path = paths[0]
        # Fetch jobs and update summary status
        summary = ExperimentSummary(
            name=experiment_path.experiment_name,
            created_at=experiment_path.experiment_creation_time,
        )
        experiment_jobs = await self._get_experiment_jobs(experiment_name)
        summary.status = self._get_experiment_status(experiment_jobs)
        # Fetch experiment result
        result = None
        if summary.status is ExperimentStatus.SUCCEEDED:
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
                job_id=job.job_id,
                additional_info=job.message,
            )
            job_summaries.append(job_summary)
        return ExperimentDetails(
            summary=summary,
            jobs=job_summaries,
            result=result,
        )


class Factory:
    @staticmethod
    def new_experiment_service(config: SurferConfig) -> ExperimentService:
        storage_client = StorageClient.from_config(config.storage)
        job_client = JobSubmissionClient(address=config.ray_address)
        return ExperimentService(
            storage_client=storage_client,
            job_client=job_client,
        )


class SurferConfigManager:
    def __init__(
        self,
        base_path=constants.SURFER_CONFIG_BASE_DIR_PATH,
        config_file_name=constants.SURFER_CONFIG_FILE_NAME,
    ):
        """
        Parameters
        ----------
        base_path: Path
            The base path to the directory containing Cloud Surfer files
        config_file_name: str
            The name of the Cloud Surfer configuration file
        """
        self.config_file_path = base_path / config_file_name

    def config_exists(self) -> bool:
        """Check if the Cloud Surfer configuration file exists

        Returns
        -------
        bool
            True if the Cloud Surfer configuration file exists,
            False otherwise
        """
        return self.config_file_path.exists()

    def save_config(self, config: SurferConfig):
        """Save to file the Cloud Surfer configuration

        If the Cloud Surfer configuration file already exists,
        it will be overwritten.

        Parameters
        ----------
        config: SurferConfig
            The Cloud Surfer configuration to save
        """
        # Create config file
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file_path, "w") as f:
            config_dict = json.loads(config.json())
            f.write(yaml.dump(config_dict))

    def load_config(self) -> Optional[SurferConfig]:
        """Load the Cloud Surfer configuration

        Raises
        ------
        InternalError
            If the Cloud Surfer file is found but failed to parse

        Returns
        -------
        Optional[SurferConfig]
            The Cloud Surfer configuration if Cloud Surfer has been
            already initialized, None otherwise
        """
        if not self.config_exists():
            return None
        try:
            with open(self.config_file_path) as f:
                config_dict = yaml.safe_load(f.read())
                return SurferConfig.parse_obj(config_dict)
        except Exception as e:
            raise InternalError(
                "error parsing CloudSurfer config at {}: {}".format(
                    self.config_file_path,
                    e,
                )
            )
