import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from surfer.core.schemas import SurferConfig
from surfer.core.services import SurferConfigManager
from surfer.storage.aws import AWSStorageConfig
from surfer.storage.azure import AzureStorageConfig
from surfer.storage.gcp import GCPStorageConfig
from surfer.storage.models import StorageConfig, StorageProvider


class TestConfigManager(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_config_exists__not_exist(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        self.assertFalse(manager.config_exists())

    def test_config_exists__exist(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        manager.config_file_path.touch()
        self.assertTrue(manager.config_exists())

    def test_load_config__not_exist(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        self.assertIsNone(manager.load_config())

    def test_load_config__invalid_config(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        with open(manager.config_file_path, "w") as f:
            f.write("foo: bar")
        with self.assertRaises(Exception):
            manager.load_config()

    def test_save_config(self):
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        cluster_file_path = Path(self.tmp_dir.name, "cluster.yaml")
        with open(cluster_file_path, "w") as f:
            f.write("")
        config = SurferConfig(
            cluster_file=cluster_file_path,
            storage=AzureStorageConfig(sas_url="https://myaccount.blob.core.windows.net/pictures"),
        )
        manager.save_config(config)
        self.assertEqual(config.json(), manager.load_config().json())

    def test_save_config__storage_serialization(self):
        # Init config file
        manager = SurferConfigManager(base_path=Path(self.tmp_dir.name))
        cluster_file_path = Path(self.tmp_dir.name, "cluster.yaml")
        with open(cluster_file_path, "w") as f:
            f.write("")
        # Available storage configs
        storage_configs = [
            AzureStorageConfig(
                sas_url="https://myaccount.blob.core.windows.net/pictures",
            ),
            GCPStorageConfig(
                project="my-project",
                bucket="my-bucket",
            ),
            AWSStorageConfig()
        ]
        # For each storage config, test whether it gets serialized/deserialized correctly
        for c in storage_configs:
            surfer_config = SurferConfig(
                cluster_file=cluster_file_path,
                storage=c,
            )
            manager.save_config(surfer_config)
            loaded_config = manager.load_config()
            self.assertEqual(c, loaded_config.storage)


class MockedStorageConfig(StorageConfig):
    provider = StorageProvider.AZURE
