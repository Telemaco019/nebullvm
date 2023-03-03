import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from pydantic.error_wrappers import ValidationError

from surfer.core.schemas import SurferConfig, ExperimentConfig
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


class TestExperimentConfig(unittest.TestCase):
    def test_validation__paths_must_resolve_to_file(self):
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValidationError):
                ExperimentConfig(
                    description="test",
                    data_loader_module=Path(tmp_dir),
                    model_loader_module=Path(tmp_dir),
                    model_evaluator_module=Path(tmp_dir),
                )

    def test_yaml_serialization_deserialization(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_file_path = Path(tmp_dir) / "tmp.py"
            with open(tmp_file_path, "w") as f:
                f.write("")
            config = ExperimentConfig(
                description="test",
                data_loader_module=tmp_file_path,
                model_loader_module=tmp_file_path,
                model_evaluator_module=tmp_file_path,
            )
            deserialized_yaml = yaml.safe_load(yaml.dump(config.dict()))
            deserialized_config = ExperimentConfig(**deserialized_yaml)
            self.assertEqual(config.dict(), deserialized_config.dict())

