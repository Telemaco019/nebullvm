import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from surfer.cli.surfer import app
from surfer.computing.clusters import Accelerator
from surfer.core.config import SurferConfigManager

runner = CliRunner()


@patch.object(SurferConfigManager, "load_config")
class TestListAccelerators(unittest.TestCase):
    @patch("surfer.computing.clusters.get_available_accelerators")
    def test_multiple_accelerators(self, mock_get_available_accelerators, _):
        mock_get_available_accelerators.return_value = [
            Accelerator.NVIDIA_TESLA_A100,
            Accelerator.NVIDIA_TESLA_V100,
        ]
        result = runner.invoke(app, ["list-accelerators"])
        self.assertEqual(result.exit_code, 0)

    @patch("surfer.computing.clusters.get_available_accelerators")
    def test_no_accelerators_found(self, mock_get_available_accelerators, _):
        mock_get_available_accelerators.return_value = []
        result = runner.invoke(app, ["list-accelerators"])
        self.assertEqual(result.exit_code, 0)
