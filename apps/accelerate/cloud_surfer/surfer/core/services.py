from typing import List, Optional

from surfer.core.config import SurferConfig
from surfer.core.models import SubmitExperimentRequest, ExperimentSummary, ExperimentDetails
from surfer.storage.clients import StorageClient


class ExperimentService:
    def __init__(self, storage_client: StorageClient):
        self.storage_client = storage_client

    async def submit(self, req: SubmitExperimentRequest):
        pass

    async def delete(self, experiment_name: str):
        pass

    async def stop(self, experiment_name: str):
        pass

    async def list(self) -> List[ExperimentSummary]:
        return []

    async def get(self, experiment_name: str) -> Optional[ExperimentDetails]:
        return None


class Factory:
    @staticmethod
    def new_experiment_service(config: SurferConfig) -> ExperimentService:
        storage_client = StorageClient.from_config(config.storage)
        return ExperimentService(storage_client=storage_client)
