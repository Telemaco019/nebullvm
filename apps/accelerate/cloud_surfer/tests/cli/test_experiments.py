import datetime
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from unittest.mock import patch, AsyncMock

import typer
from typer.testing import CliRunner

from surfer.cli.commands.common import must_load_config
from surfer.cli.experiments import app
from surfer.common import exceptions
from surfer.common.schemas import (
    SurferConfig,
    ExperimentResult,
    OptimizedModelDescriptor,
)
from surfer.core.config import SurferConfigManager
from surfer.core.experiments import (
    ExperimentStatus,
    ExperimentSummary,
    JobSummary,
    ExperimentDetails,
)
from tests import test_utils
from tests.test_utils import MockedStorageConfig, new_optimization_result

runner = CliRunner()


class TestMustLoadConfig(unittest.TestCase):
    @patch.object(SurferConfigManager, "load_config")
    def test_config_not_found(self, load_config_mock):
        load_config_mock.return_value = None
        with self.assertRaises(typer.Exit):
            must_load_config()

    @patch.object(SurferConfigManager, "load_config")
    def test_config_found(self, load_config_mock):
        with NamedTemporaryFile() as f:
            f.write(b"")
            config = SurferConfig(
                cluster_file=Path(f.name),
                storage=MockedStorageConfig(),
                ray_address="https://localhost:8265",
            )
            load_config_mock.return_value = config
            self.assertEqual(config, must_load_config())


@patch("surfer.core.experiments.new_experiment_service")
@patch.object(SurferConfigManager, "load_config")
class TestExperimentCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.cluster_file_path = Path(self.tmp_dir.name) / "cluster.yaml"
        with open(self.cluster_file_path, "w") as f:
            f.write("")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _new_mocked_config(self) -> SurferConfig:
        return SurferConfig(
            cluster_file=self.cluster_file_path,
            storage=MockedStorageConfig(),
        )

    def test_list_experiments__empty_list(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.list.return_value = []
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, "list")
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_list_experiments__multiple_experiments(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.list.return_value = [
            ExperimentSummary(
                name="test",
                created_at=datetime.datetime.now(),
                status=ExperimentStatus.UNKNOWN,
            ),
            ExperimentSummary(
                name="test",
                created_at=datetime.datetime.now(),
                status=ExperimentStatus.RUNNING,
            ),
        ]
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, "list")
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_describe_experiment__not_found(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.get.return_value = None
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["describe", "test"])
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_describe_experiment__no_results_no_jobs(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()

        service_mock = AsyncMock()
        service_mock.get.return_value = ExperimentDetails(
            summary=ExperimentSummary(
                name="test",
                created_at=datetime.datetime.now(),
                status=ExperimentStatus.UNKNOWN,
            ),
            jobs=[],
            result=None,
        )
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["describe", "test"])
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_describe_experiment__internal_error(
        self,
        _,
        factory,
        *__,
    ):
        service_mock = AsyncMock()
        service_mock.get.side_effect = exceptions.InternalError
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["describe", "test"])
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_describe_experiment__no_results(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.get.return_value = ExperimentDetails(
            summary=ExperimentSummary(
                name="test",
                created_at=datetime.datetime.now(),
                status=ExperimentStatus.UNKNOWN,
            ),
            jobs=[
                JobSummary(
                    job_id="test-1",
                    status="unknown",
                ),
                JobSummary(
                    job_id="test-2",
                    status="unknown",
                ),
            ],
            result=None,
        )
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["describe", "test"])
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_describe_experiment__result_with_no_optimizations(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.get.return_value = ExperimentDetails(
            summary=ExperimentSummary(
                name="test",
                created_at=datetime.datetime.now(),
                status=ExperimentStatus.UNKNOWN,
            ),
            jobs=[
                JobSummary(
                    job_id="test-1",
                    status="unknown",
                ),
                JobSummary(
                    job_id="test-2",
                    status="unknown",
                ),
            ],
            result=ExperimentResult(
                optimizations=[],
            ),
        )
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["describe", "test"])
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_describe_experiment__results_without_optimized_models(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.get.return_value = ExperimentDetails(
            summary=ExperimentSummary(
                name="test",
                created_at=datetime.datetime.now(),
                status=ExperimentStatus.UNKNOWN,
            ),
            jobs=[
                JobSummary(
                    job_id="test-1",
                    status="unknown",
                ),
                JobSummary(
                    job_id="test-2",
                    status="unknown",
                ),
            ],
            result=ExperimentResult(
                optimizations=[
                    new_optimization_result(
                        optimized_model=None,
                    ),
                    new_optimization_result(
                        optimized_model=OptimizedModelDescriptor(
                            latency_seconds=0.9,
                            throughput=0,
                            size_mb=0,
                            technique="",
                            compiler="",
                            metric_drop=0,
                            model_path=Path(),
                        ),
                        latency_rate_improvement=1,
                        throughput_rate_improvement=1,
                        size_rate_improvement=1,
                    ),
                ],
            ),
        )
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["describe", "test"])
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_stop_experiment__should_abort_without_confirm(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.stop.side_effect = exceptions.NotFoundError
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["stop", "test"])
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_stop_experiment__not_found(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.stop.side_effect = exceptions.NotFoundError
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["stop", "test"], input="y")
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_stop_experiment__internal_error(
        self,
        _,
        factory,
        *__,
    ):
        service_mock = AsyncMock()
        service_mock.stop.side_effect = exceptions.InternalError
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["stop", "test"], input="y")
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_stop_experiment__value_error(
        self,
        _,
        factory,
        *__,
    ):
        service_mock = AsyncMock()
        service_mock.stop.side_effect = ValueError
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["stop", "test"], input="y")
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_stop_experiment__success(
        self,
        _,
        factory,
        *__,
    ):
        service_mock = AsyncMock()
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["stop", "test"], input="y")
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_delete_experiment__should_abort_without_confirm(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.delete.side_effect = exceptions.NotFoundError
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["delete", "test"])
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_delete_experiment__not_found(
        self,
        load_config_mock,
        factory,
        *_,
    ):
        load_config_mock.return_value = self._new_mocked_config()
        service_mock = AsyncMock()
        service_mock.delete.side_effect = exceptions.NotFoundError
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["delete", "test"], input="y")
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_delete_experiment__internal_error(
        self,
        _,
        factory,
        *__,
    ):
        service_mock = AsyncMock()
        service_mock.delete.side_effect = exceptions.InternalError
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["delete", "test"], input="y")
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_delete_experiment__success(
        self,
        _,
        factory,
        *__,
    ):
        service_mock = AsyncMock()
        factory.return_value = service_mock
        # Run command
        result = runner.invoke(app, ["delete", "test"], input="y")
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_submit_experiment__config_path_does_not_exist(self, *_):
        # Run command
        result = runner.invoke(
            app,
            ["submit", "/tmp/invalid.yaml", "--name", "test"],
        )
        self.assertEqual(2, result.exit_code, result.stdout)

    def test_submit_experiment__config_path_is_dir(self, *_):
        with TemporaryDirectory() as tmpdir:
            # Run command
            result = runner.invoke(app, ["submit", tmpdir, "--name", "test"])
            self.assertEqual(2, result.exit_code, result.stdout)

    def test_submit_experiment__success(
        self,
        _,
        factory,
        *__,
    ):
        service_mock = AsyncMock()
        factory.return_value = service_mock
        # Run command
        with test_utils.tmp_experiment_config_file() as tmp_file:
            result = runner.invoke(
                app,
                ["submit", tmp_file.as_posix(), "--name", "test"],
            )
        self.assertEqual(0, result.exit_code, result.stdout)

    def test_submit_experiment__internal_error(
        self,
        _,
        factory,
        *__,
    ):
        service_mock = AsyncMock()
        factory.return_value = service_mock
        service_mock.submit.side_effect = exceptions.InternalError
        # Run command
        with test_utils.tmp_experiment_config_file() as tmp_file:
            result = runner.invoke(
                app,
                ["submit", tmp_file.as_posix(), "--name", "test"],
            )
        self.assertEqual(1, result.exit_code, result.stdout)

    def test_submit_experiment__config_deserialization_error(self, *_):
        with TemporaryDirectory() as d:
            tmp_file = Path(d, "config.yaml")
            with open(tmp_file, "w") as f:
                f.write("invalid: config")
            result = runner.invoke(
                app,
                ["submit", tmp_file.as_posix(), "--name", "test"],
            )
        self.assertEqual(1, result.exit_code, result.stdout)
