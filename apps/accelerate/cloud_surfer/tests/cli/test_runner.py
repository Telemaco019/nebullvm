import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

from surfer.cli.runner import RunCommandBuilder
from surfer.cli.runner import app
from surfer.common.schemas import SurferConfig
from surfer.computing.models import Accelerator
from surfer.storage import AzureStorageConfig
from tests import _get_assets_path

runner = CliRunner()


class TestRun(unittest.TestCase):
    @patch("surfer.cli.commands.cmd_run.run")
    def test_all_options(self, mocked_run):
        ignored_accelerators = [
            Accelerator.NVIDIA_TESLA_K80,
            Accelerator.NVIDIA_TESLA_V100,
        ]
        ignored_compilers = [
            "test",
        ]
        with TemporaryDirectory() as tmp:
            surfer_config = SurferConfig(
                cluster_file=_get_assets_path() / Path("empty.py"),
                storage=AzureStorageConfig(
                    sas_url="https://myaccount.blob.core.windows.net/pictures"
                ),
                ray_address="http://127.0.0.1",
            )
            surfer_config_path = Path(tmp) / Path("surfer_config.yaml")
            with open(surfer_config_path, "w") as f:
                f.write(surfer_config.json())
            command = (
                RunCommandBuilder()
                .with_model_loader(_get_assets_path() / Path("model_loaders.py"))
                .with_data_loader(_get_assets_path() / Path("data_loaders.py"))
                .with_model_evaluator(_get_assets_path() / Path("model_evaluators.py"))
                .with_surfer_config(surfer_config_path)
                .with_results_dir(_get_assets_path())
                .with_metric_drop_threshold(0)
                .with_ignored_accelerators(ignored_accelerators)
                .with_ignored_compilers(ignored_compilers)
                .with_debug()
                .get_command()
            )
            command_parts = command.split(" ")
            res = runner.invoke(app, command_parts[3:])
        self.assertEqual(0, res.exit_code, res.stdout)
        mocked_run.assert_called_once()
