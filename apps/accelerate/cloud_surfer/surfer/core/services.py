import asyncio
import json
from pathlib import Path
from typing import List, Optional

import yaml
from ray.job_submission import JobDetails, JobStatus
from ray.job_submission import JobSubmissionClient

from surfer.core import constants
from surfer.core.models import SubmitExperimentRequest, ExperimentSummary, ExperimentDetails, ExperimentStatus
from surfer.core.schemas import SurferConfig
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

    async def list(self) -> List[ExperimentSummary]:
        """List all experiments

        List all the available experiments for which there is any data saved in the storage.

        Returns
        -------
        List[ExperimentSummary]
            Summary containing essential information of each available experiment
        """
        # Fetch experiments data from storage
        paths = await self.storage_client.list(f"{constants.EXPERIMENTS_STORAGE_PREFIX}/")
        summaries = []
        for path in paths:
            try:
                summaries.append(ExperimentSummary.from_path(path))
            except Exception:
                pass
        # No experiment data is found, we are done
        if len(summaries) == 0:
            return summaries
        # Fetch experiments status and update summaries
        jobs = await asyncio.get_event_loop().run_in_executor(None, self.job_client.list_jobs)
        for summary in summaries:
            experiment_jobs = self._filter_experiment_jobs(jobs, summary.name)
            summary.status = self._get_experiment_status(experiment_jobs)
        return summaries

    async def get(self, experiment_name: str) -> Optional[ExperimentDetails]:
        return None


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
