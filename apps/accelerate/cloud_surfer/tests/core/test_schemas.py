import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from surfer.core.schemas import SurferConfig
from tests.core.test_services import MockedStorageConfig


class TestSurferConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_cluster_file__path_not_exist(self):
        with self.assertRaises(Exception):
            SurferConfig(
                cluster_file=Path("/invalid/path/cluster.yaml"),
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
