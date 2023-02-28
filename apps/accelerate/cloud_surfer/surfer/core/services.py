from typing import List

from surfer.core.models import SubmitExperimentRequest, ExperimentSummary
from surfer.storage.clients import StorageClient


class ExperimentService:
    def __init__(self, storage_client: StorageClient):
        self.storage_client = storage_client

    def submit(self, req: SubmitExperimentRequest):
        pass

    def delete(self, experiment_name: str):
        pass

    def stop(self, experiment_name: str):
        pass

    def list(self) -> List[ExperimentSummary]:
        pass
