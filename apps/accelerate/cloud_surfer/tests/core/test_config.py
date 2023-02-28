import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from surfer.core.config import SurferConfigManager, SurferConfig
from surfer.storage.models import StorageConfig, StorageProvider


class MockedStorageConfig(StorageConfig):
    provider = StorageProvider.AZURE


class TestSurferConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_cluster_file__path_not_exist(self):
        with self.assertRaises(Exception):
            SurferConfig(
                cluster_file=Path("/invalid/path"),
                storage=MockedStorageConfig(),
            )

    def test_cluster_file__path_not_a_file(self):
        with self.assertRaises(Exception):
            SurferConfig(
                cluster_file=Path(self.tmp_dir.name),
                storage=MockedStorageConfig(),
            )

    def test_cluster_file__invalid_yaml(self):
        invalid_yaml_path = Path(self.tmp_dir.name) / "invalid.yaml"
        with open(invalid_yaml_path, "w") as f:
            f.write("{")
        with self.assertRaises(yaml.YAMLError):
            SurferConfig(
                cluster_file=invalid_yaml_path,
                storage=MockedStorageConfig(),
            )

    def test_cluster_file__valid_yaml(self):
        valid_yaml_path = Path(self.tmp_dir.name) / "valid.yaml"
        with open(valid_yaml_path, "w") as f:
            f.write("foo: bar")
        SurferConfig(
            cluster_file=valid_yaml_path,
            storage=MockedStorageConfig(),
        )


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
            storage=MockedStorageConfig(),
        )
        manager.save_config(config)
        self.assertEqual(config, manager.load_config())
