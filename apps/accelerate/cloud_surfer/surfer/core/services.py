import asyncio
import json
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic.error_wrappers import ValidationError
from ray.job_submission import JobDetails, JobStatus
from ray.job_submission import JobSubmissionClient

from surfer.core import constants
from surfer.core.exceptions import InternalError
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


class ExperimentService:
    def __init__(self, storage_client: StorageClient, job_client: JobSubmissionClient):
        self.storage_client = storage_client
        self.job_client = job_client

    @staticmethod
    def _filter_experiment_jobs(jobs: List[JobDetails], experiment_name: str) -> List[JobDetails]:
        return [j for j in jobs if experiment_name == j.metadata.get(constants.JOB_METADATA_EXPERIMENT_NAME, None)]

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
        # Default - the experiment status is unknown
        return ExperimentStatus.UNKNOWN

    async def submit(self, req: SubmitExperimentRequest):
        pass

    async def delete(self, experiment_name: str):
        pass

    async def stop(self, experiment_name: str):
        pass

    async def _fetch_all_jobs(self) -> List[JobDetails]:
        return await asyncio.get_event_loop().run_in_executor(None, self.job_client.list_jobs)

    async def _get_experiment_paths(self, experiment_name: Optional[str] = None) -> List[ExperimentPath]:
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

    async def list(self) -> List[ExperimentSummary]:
        """List all experiments

        List all the available experiments for which there is any data saved in the storage.

        Returns
        -------
        List[ExperimentSummary]
            Summary containing essential information of each available experiment
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

    async def _fetch_result(self, path: ExperimentPath) -> Optional[ExperimentResult]:
        try:
            raw_data = await self.storage_client.get(path.as_path())
            if raw_data is None:
                return None
            return ExperimentResult.parse_raw(raw_data)
        except ValidationError as e:
            raise InternalError(f"Failed to parse experiment result: {e}")

    async def get(self, experiment_name: str) -> Optional[ExperimentDetails]:
        """

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
        if len(paths) > 1:
            logger.warn(f"Found multiple entries for experiment {experiment_name} - using first")
        experiment_path = paths[0]
        # Fetch jobs and update summary status
        summary = ExperimentSummary(
            name=experiment_path.experiment_name,
            created_at=experiment_path.experiment_creation_time,
        )
        jobs = await self._fetch_all_jobs()
        experiment_jobs = self._filter_experiment_jobs(jobs, summary.name)
        summary.status = self._get_experiment_status(experiment_jobs)
        # Fetch experiment result
        result = None
        if summary.status is ExperimentStatus.SUCCEEDED:
            result = await self._fetch_result(experiment_path)
            if result is None:
                logger.warn(f"Experiment {experiment_name} is succeeded, but results are missing")
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
        return ExperimentService(storage_client=storage_client, job_client=job_client)


class SurferConfigManager:
    def __init__(
        self,
        base_path: Path = constants.SURFER_CONFIG_BASE_DIR_PATH,
        config_file_name=constants.SURFER_CONFIG_FILE_NAME,
    ):
        self.config_file_path = base_path / config_file_name

    def config_exists(self) -> bool:
        return self.config_file_path.exists()

    def save_config(self, config: SurferConfig):
        # Create config file
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file_path, "w") as f:
            config_dict = json.loads(config.json())
            f.write(yaml.dump(config_dict))

    def load_config(self) -> Optional[SurferConfig]:
        if not self.config_exists():
            return None
        try:
            with open(self.config_file_path) as f:
                config_dict = yaml.safe_load(f.read())
                return SurferConfig.parse_obj(config_dict)
        except Exception as e:
            raise Exception(f"Error parsing CloudSurfer config at {self.config_file_path}: {e}")
