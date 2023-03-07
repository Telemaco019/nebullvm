import datetime
import unittest
from pathlib import Path
from tempfile import mkstemp

import yaml

from surfer.core import models, constants
from tests.core.test_services import MockedStorageConfig


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
        self.assertEqual(experiment_name, experiment_path.experiment_name)
        self.assertEqual(
            creation_date, experiment_path.experiment_creation_time
        )


class TestJobWorkingDir(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.paths_to_cleanup = []

    def tearDown(self) -> None:
        for path in self.paths_to_cleanup:
            path.unlink()

    async def test_all_files_are_copied(self):
        _, cluster_file = mkstemp()
        cluster_file_path = Path(cluster_file)
        self.paths_to_cleanup.append(cluster_file_path)

        _, data_loader_file = mkstemp()
        data_loader_path = Path(data_loader_file)
        self.paths_to_cleanup.append(data_loader_path)

        _, model_loader_file = mkstemp()
        model_loader_path = Path(model_loader_file)
        self.paths_to_cleanup.append(model_loader_path)

        _, model_evaluator_file = mkstemp()
        model_evaluator_path = Path(model_evaluator_file)
        self.paths_to_cleanup.append(model_evaluator_path)

        surfer_config = models.SurferConfig(
            cluster_file=cluster_file_path,
            storage=MockedStorageConfig(),
        )
        original_surfer_config = surfer_config.copy()
        experiment_config = models.ExperimentConfig(
            data_loader_module=data_loader_path,
            model_loader_module=model_loader_path,
            model_evaluator_module=model_evaluator_path,
        )

        async with models.job_working_dir(
            surfer_config,
            experiment_config,
        ) as workdir:
            self.assertTrue(
                (workdir.base / workdir.surfer_config_path).exists()
            )

            self.assertTrue(
                (workdir.base / workdir.data_loader_path).exists()
            )
            self.assertNotEqual(data_loader_path, workdir.data_loader_path)

            self.assertTrue(
                (workdir.base / workdir.model_loader_path).exists()
            )
            self.assertNotEqual(model_loader_path, workdir.model_loader_path)

            self.assertTrue(
                (workdir.base / workdir.model_evaluator_path).exists()
            )
            self.assertNotEqual(
                model_evaluator_path, workdir.model_evaluator_path
            )

            # Check cluster file has been copied
            with open(workdir.base / workdir.surfer_config_path) as f:
                obj = yaml.safe_load(f.read())
                loaded = models.SurferConfig.parse_obj(obj)
                self.assertTrue(loaded.cluster_file.exists())
                self.assertNotEqual(
                    loaded.cluster_file, cluster_file_path
                )

        # Check surfer config is not modified
        self.assertEqual(original_surfer_config, surfer_config)
