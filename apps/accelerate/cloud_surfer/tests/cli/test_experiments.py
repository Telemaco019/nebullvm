import unittest
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from surfer.cli.experiments import app, _must_load_config
from surfer.core import exceptions
from surfer.core.config import SurferConfigManager, SurferConfig
from surfer.core.services import ExperimentService
from tests.core.test_config import MockedStorageConfig

runner = CliRunner()


class TestMustGetConfig(unittest.TestCase):
    @patch.object(SurferConfigManager, "load_config")
    def test_config_not_found(self, load_config_mock):
        load_config_mock.return_value = None
        with self.assertRaises(typer.Exit):
            _must_load_config()

    @patch.object(SurferConfigManager, "load_config")
    def test_config_found(self, load_config_mock):
        with NamedTemporaryFile() as f:
            f.write(b"")
            config = SurferConfig(
                cluster_file=Path(f.name),
                storage=MockedStorageConfig(),
            )
            load_config_mock.return_value = config
            self.assertEqual(config, _must_load_config())


@patch.object(SurferConfigManager, "load_config")
class TestExperimentCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.cluster_file_path = Path(self.tmp_dir.name) / "cluster.yaml"
        with open(self.cluster_file_path, "w") as f:
            f.write("")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _new_mocked_config(self)->SurferConfig:
        return SurferConfig(
            cluster_file=self.cluster_file_path,
            storage=MockedStorageConfig(),
        )

    @patch.object(ExperimentService, "list")
    def test_list_experiments__empty_list(
            self,
            list_experiments_mock,
            load_config_mock,
            *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        list_experiments_mock.return_value = []
        # Run command
        result = runner.invoke(app, "list")
        self.assertEqual(0, result.exit_code)

    @patch.object(ExperimentService, "get")
    def test_describe_experiment__not_found(
            self,
            get_experiment_mock,
            load_config_mock,
            *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        get_experiment_mock.return_value = None
        # Run command
        result = runner.invoke(app, ["describe", "test"])
        self.assertEqual(1, result.exit_code)

    @patch.object(ExperimentService, "stop")
    def test_stop_experiment__not_found(
            self,
            stop_experiment_mock,
            load_config_mock,
            *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        stop_experiment_mock.side_effect = exceptions.NotFoundError
        # Run command
        result = runner.invoke(app, ["stop", "test"])
        self.assertEqual(1, result.exit_code)

    @patch.object(ExperimentService, "delete")
    def test_delete_experiment__not_found(
            self,
            delete_experiment_mock,
            load_config_mock,
            *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        delete_experiment_mock.side_effect = exceptions.NotFoundError
        # Run command
        result = runner.invoke(app, ["delete", "test"])
        self.assertEqual(1, result.exit_code)
