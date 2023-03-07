import unittest
from pathlib import Path

from surfer.runner.services import SpeedsterResultsCollector
from tests import _get_assets_path


class TestSpeedsterResultsCollector(unittest.TestCase):
    def test_collect_results__files_dir_does_not_exist(self):
        results_collector = SpeedsterResultsCollector(
            result_files_dir=Path("/tmp/unexisting")
        )
        with self.assertRaises(ValueError):
            results_collector.collect_results()

    def test_collect_results__multiple_files(self):
        path = _get_assets_path() / "latencies"
        results_collector = SpeedsterResultsCollector(result_files_dir=path)
        result = results_collector.collect_results()
        self.assertGreater(len(result), 1)
