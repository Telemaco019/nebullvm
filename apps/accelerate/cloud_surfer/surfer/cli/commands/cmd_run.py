from surfer.common.schemas import SurferConfig
from surfer.core.clusters import get_ray_cluster
from surfer.core.orchestrators import RunConfig, RayOrchestrator
from surfer.storage.clients import StorageClient


def run(
    surfer_config: SurferConfig,
    run_config: RunConfig,
    experiment_name: str
):
    cluster = get_ray_cluster(surfer_config)
    storage_client = StorageClient.from_config(surfer_config.storage)
    orchestrator = RayOrchestrator(
        cluster=cluster,
        storage_client=storage_client,
    )
    orchestrator.run_experiment(experiment_name, run_config)
