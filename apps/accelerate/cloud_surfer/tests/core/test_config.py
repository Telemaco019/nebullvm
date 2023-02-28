import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from surfer.core.config import SurferConfigManager, SurferConfig


class TestSurferConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_cluster_file__path_not_exist(self):
        with self.assertRaises(Exception):
            SurferConfig(cluster_file=Path("/invalid/path"))

    def test_cluster_file__path_not_a_file(self):
        with self.assertRaises(Exception):
            SurferConfig(cluster_file=Path(self.tmp_dir.name))

    def test_cluster_file__invalid_yaml(self):
        invalid_yaml_path = Path(self.tmp_dir.name) / "invalid.yaml"
        with open(invalid_yaml_path, "w") as f:
            f.write("{")
        with self.assertRaises(yaml.YAMLError):
            SurferConfig(cluster_file=invalid_yaml_path)

    def test_cluster_file__valid_yaml(self):
        valid_yaml_path = Path(self.tmp_dir.name) / "valid.yaml"
        with open(valid_yaml_path, "w") as f:
            f.write("foo: bar")
        SurferConfig(cluster_file=valid_yaml_path)


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
