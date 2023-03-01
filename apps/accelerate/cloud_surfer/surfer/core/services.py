from typing import List, Optional

from ray.job_submission import JobSubmissionClient

from surfer.core import constants
from surfer.core.config import SurferConfig
from surfer.core.models import SubmitExperimentRequest, ExperimentSummary, ExperimentDetails
from surfer.log import logger
from surfer.storage.clients import StorageClient


class ExperimentService:
    def __init__(self, storage_client: StorageClient, job_client: JobSubmissionClient):
        self.storage_client = storage_client
        self.job_client = job_client

    async def submit(self, req: SubmitExperimentRequest):
        pass

    async def delete(self, experiment_name: str):
        pass

    async def stop(self, experiment_name: str):
        pass

    async def list(self) -> List[ExperimentSummary]:
        # Fetch experiments data from storage
        paths = await self.storage_client.list(f"{constants.EXPERIMENTS_STORAGE_PREFIX}/*")
        summaries = []
        for path in paths:
            try:
                summaries.append(ExperimentSummary.from_path(path))
            except ValueError:
                logger.warn(f"Found invalid path in experiments storage: {path}")
        # Fetch experiment running status TODO
        # client = JobSubmissionClient("http://127.0.0.1:8265")
        # for j in client.list_jobs():
        #     j.status
        return summaries

    async def get(self, experiment_name: str) -> Optional[ExperimentDetails]:
        return None


class Factory:
    @staticmethod
    def new_experiment_service(config: SurferConfig) -> ExperimentService:
        storage_client = StorageClient.from_config(config.storage)
        job_client = JobSubmissionClient(address=config.ray_address)
        return ExperimentService(storage_client=storage_client, job_client=job_client)
