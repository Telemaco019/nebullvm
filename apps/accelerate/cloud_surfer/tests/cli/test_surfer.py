import unittest
from tempfile import NamedTemporaryFile
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
        self.assertEqual(result.exit_code, 0, result.stdout)

    @patch("surfer.computing.clusters.get_available_accelerators")
    def test_no_accelerators_found(self, mock_get_available_accelerators, _):
        mock_get_available_accelerators.return_value = []
        result = runner.invoke(app, ["list-accelerators"])
        self.assertEqual(result.exit_code, 0, result.stdout)


class TestInitSurferConfigCli(unittest.TestCase):
    def test_storage_provider_azure__invalid_url(self):
        with NamedTemporaryFile() as tmp:
            # Run command
            args = [
                "init",
                tmp.name,
                "--storage-provider",
                "azure",
                "--ray-address",
                "127.0.0.1",
            ]
            result = runner.invoke(
                app,
                args,
                input="https://foo.bar\n",
            )
            self.assertEqual(1, result.exit_code, result.stdout)

    @patch("surfer.storage.enabled_providers", new=[])
    def test_storage_provider_azure__provider_not_enabled(self):
        with NamedTemporaryFile() as tmp:
            # Run command
            args = [
                "init",
                tmp.name,
                "--storage-provider",
                "azure",
                "--ray-address",
                "127.0.0.1",
            ]
            result = runner.invoke(
                app,
                args,
                input="https://myaccount.blob.core.windows.net/pictures\n",
            )
            self.assertEqual(1, result.exit_code, result.stdout)

    @patch.object(SurferConfigManager, "config_exists", return_value=False)
    @patch.object(SurferConfigManager, "save_config")
    def test_storage_provider_azure__valid_url(
        self,
        *_,
    ):
        with NamedTemporaryFile() as tmp:
            # Run command
            args = [
                "init",
                tmp.name,
                "--storage-provider",
                "azure",
                "--ray-address",
                "http://127.0.0.1",
            ]
            result = runner.invoke(
                app,
                args,
                input="https://myaccount.blob.core.windows.net/pictures\n",
            )
            self.assertEqual(0, result.exit_code, result.stdout)

    @patch("surfer.storage.enabled_providers", new=[])
    def test_storage_provider_gcp__provider_not_enabled(self):
        with NamedTemporaryFile() as tmp:
            # Run command
            args = [
                "init",
                tmp.name,
                "--storage-provider",
                "gcp",
            ]
            result = runner.invoke(
                app,
                args,
            )
            self.assertEqual(1, result.exit_code, result.stdout)

    @patch("surfer.storage.enabled_providers", new=[])
    def test_storage_provider_aws__provider_not_enabled(self):
        with NamedTemporaryFile() as tmp:
            # Run command
            args = [
                "init",
                tmp.name,
                "--storage-provider",
                "aws",
            ]
            result = runner.invoke(
                app,
                args,
            )
            self.assertEqual(1, result.exit_code, result.stdout)

    @patch.object(SurferConfigManager, "config_exists", return_value=True)
    @patch.object(SurferConfigManager, "save_config")
    @patch.object(SurferConfigManager, "load_config")
    def test_config_exists_should_ask_confirm(
        self,
        *_,
    ):
        with NamedTemporaryFile() as tmp:
            args = [
                "init",
                tmp.name,
                "--storage-provider",
                "azure",
                "--ray-address",
                "http://127.0.0.1",
            ]
            # Run command without confirm -- should exit with 1
            result = runner.invoke(
                app,
                args,
                input="https://myaccount.blob.core.windows.net/p\n",
            )
            self.assertEqual(1, result.exit_code, result.stdout)

            # Run command with confirm -- should exit with 0
            result = runner.invoke(
                app,
                args,
                input="https://myaccount.blob.core.windows.net/p\ny\n",
            )
            self.assertEqual(0, result.exit_code, result.stdout)
