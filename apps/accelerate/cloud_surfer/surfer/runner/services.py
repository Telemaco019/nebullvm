import json
from pathlib import Path
from typing import Dict

from surfer.log import logger
from surfer.runner.models import RayCluster


class SpeedsterResultsCollector:
    def __init__(
        self,
        result_files_dir: Path = Path("."),
        result_files_regex: str = "*.json",
    ):
        """
        Parameters
        ----------
        result_files_dir: Path
            The path to the dir containing the results file
            produced by Speedster that will be collected
        result_files_regex: Path
            The regex for finding the results files produces by Speedster
        """
        self._results_file_dir = result_files_dir
        self._results_file_regex = result_files_regex

    def collect_results(self) -> Dict[str, any]:
        """Collect the results of a single Speedster run

        Returns
        -------
        Dict[str, any]
            A dictionary containing the results produced by Speedster
        """
        logger.info("collecting Nebullvm results...")
        result_riles = [
            f for f in self._results_file_dir.glob(self._results_file_regex)
        ]
        if len(result_riles) == 0:
            msg = "could not find any Nebullvm results file in path {}".format(
                self._results_file_dir
            )
            raise ValueError(msg)
        if len(result_riles) > 1:
            logger.warn(
                f"found {len(result_riles)} Nebullvm results file, "
                f"using only {result_riles[0]}"
            )
        with open(
            result_riles[0],
            "r",
        ) as res_file:
            return json.load(res_file)


class Orchestrator:
    def __init__(self, cluster: RayCluster):
        self.cluster = cluster


class ModelOptimizer:
    def __init__(
        self,
        storage_client,
        model_loader,
        data_loader,
        model_evaluator,
    ):
        self.storage_client = storage_client

    def optimize(self) -> dict:
        optimize_model()
        return {}
