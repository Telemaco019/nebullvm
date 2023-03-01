import datetime
import unittest
from pathlib import Path

from surfer.core import models, constants
from surfer.core.models import ExperimentStatus


class TestExperimentSummary(unittest.TestCase):
    def test_from_path__invalid_prefix_path(self):
        with self.assertRaises(ValueError):
            models.ExperimentSummary.from_path(Path("invalid/path"))

    def test_from_path__valid_prefix_wrong_format(self):
        with self.assertRaises(ValueError):
            models.ExperimentSummary.from_path(Path(f"{constants.EXPERIMENTS_STORAGE_PREFIX}/invalid/path"))

    def test_from_path__valid_path(self):
        experiment_name = "test"
        creation_date = datetime.datetime.now().strftime(constants.INTERNAL_DATETIME_FORMAT)
        path = Path(
            constants.EXPERIMENTS_STORAGE_PREFIX,
            creation_date,
            experiment_name,
        )
        summary = models.ExperimentSummary.from_path(path)
        self.assertEqual(experiment_name, summary.name)
        self.assertEqual(creation_date, summary.created_at.strftime(constants.INTERNAL_DATETIME_FORMAT))
        self.assertEqual(ExperimentStatus.UNKNOWN, summary.status)
