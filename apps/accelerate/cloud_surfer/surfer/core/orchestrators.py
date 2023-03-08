from pathlib import Path
from typing import List, Optional

from surfer.common import constants
from surfer.storage.clients import StorageClient


class RayCluster:
    def __init__(self, config: dict):
        self.config = config

    def get_available_accelerators(self) -> List[str]:
        accelerators = []
        for node in self.config["available_node_types"]:
            resources = node.get("resources", None)
            if resources is None:
                continue
            for r in resources:
                parts = r.split(constants.CLUSTER_ACCELERATOR_TYPE_PREFIX)
                if len(parts) > 1:
                    accelerators.append(parts[1])
        return accelerators


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
