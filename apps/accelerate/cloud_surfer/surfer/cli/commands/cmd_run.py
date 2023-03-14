from pathlib import Path

from surfer.common.schemas import SurferConfig
from surfer.computing.clusters import get_ray_cluster
from surfer.core.orchestrators import RunConfig, RayOrchestrator
from surfer.storage.clients import StorageClient


def run(surfer_config: SurferConfig, run_config: RunConfig, results_dir: Path):
    """

    Parameters
    ----------
    surfer_config
    run_config
    results_dir: Path
        Remote path on the cloud storage where
        experiment results data will be stored.

    Returns
    -------

    """
    cluster = get_ray_cluster(surfer_config)
    storage_client = StorageClient.from_config(surfer_config.storage)
    orchestrator = RayOrchestrator(
        cluster=cluster,
        storage_client=storage_client,
        surfer_config=surfer_config,
    )
    orchestrator.run_experiment(results_dir, run_config)
