from pathlib import Path
from typing import Optional

from surfer.core.clusters import RayCluster
from surfer.storage.clients import StorageClient


class OptimizationConfig:
    model_loader: Path
    data_loader: Path
    model_evaluator: Optional[Path]


class OrchestrationService:
    def __init__(
        self,
        cluster: RayCluster,
        storage_client: StorageClient,
    ):
        self.cluster = cluster
        self.storage_client = storage_client

    def run_experiment(self, experiment: str, config: OptimizationConfig):
        pass
