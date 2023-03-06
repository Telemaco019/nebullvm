import datetime
import unittest
from pathlib import Path

from surfer.core import models, constants


class TestExperimentPath(unittest.TestCase):
    def test_from_path__invalid_prefix_path(self):
        with self.assertRaises(ValueError):
            models.ExperimentPath.from_path(Path("invalid/path"))

    def test_from_path__valid_prefix_wrong_format(self):
        with self.assertRaises(ValueError):
            models.ExperimentPath.from_path(
                Path(f"{constants.EXPERIMENTS_STORAGE_PREFIX}/invalid/path"),
            )

    def test_from_path__valid_path(self):
        experiment_name = "test"
        creation_date = datetime.datetime.now()
        path = models.ExperimentPath(
            experiment_name=experiment_name,
            experiment_creation_time=creation_date,
        ).as_path()
        experiment_path = models.ExperimentPath.from_path(path)
        self.assertEqual(
            experiment_name, experiment_path.experiment_name
        )
        self.assertEqual(
            creation_date, experiment_path.experiment_creation_time
        )
