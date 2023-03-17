import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import patch

import yaml
from pydantic.error_wrappers import ValidationError

from surfer.common.schemas import (
    SurferConfig,
    ExperimentConfig,
    OptimizationResult,
    HardwareInfo,
    OriginalModelDescriptor,
    OptimizedModelDescriptor,
)
from surfer.computing.models import VMProvider
from surfer.storage import (
    StorageProvider,
    AzureStorageConfig,
    GCPStorageConfig,
    AWSStorageConfig,
)
from tests.test_utils import MockedStorageConfig


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

    @patch(
        "surfer.storage.enabled_providers",
        [StorageProvider.GCP, StorageProvider.AWS],
    )
    def test_azure_provider_not_enabled(self):
        with open(Path(self.tmp_dir.name, "tmp"), "wb") as f:
            f.write(b"")
            surfer_config = SurferConfig(
                cluster_file=Path(f.name),
                storage=AzureStorageConfig(
                    sas_url="",
                ),
            )
            obj = surfer_config.dict()
            with self.assertRaises(ValueError):
                SurferConfig.parse_obj(obj)

    @patch(
        "surfer.storage.enabled_providers",
        [StorageProvider.AZURE, StorageProvider.AWS],
    )
    def test_gcp_provider_not_enabled(self):
        with open(Path(self.tmp_dir.name, "tmp"), "wb") as f:
            f.write(b"")
            surfer_config = SurferConfig(
                cluster_file=Path(f.name),
                storage=GCPStorageConfig(
                    project="",
                    bucket="",
                ),
            )
            obj = surfer_config.dict()
            with self.assertRaises(ValueError):
                SurferConfig.parse_obj(obj)

    @patch(
        "surfer.storage.enabled_providers",
        [StorageProvider.AZURE, StorageProvider.GCP],
    )
    def test_aws_provider_not_enabled(self):
        with open(Path(self.tmp_dir.name, "tmp"), "wb") as f:
            f.write(b"")
            surfer_config = SurferConfig(
                cluster_file=Path(f.name),
                storage=AWSStorageConfig(),
            )
            obj = surfer_config.dict()
            with self.assertRaises(ValueError):
                SurferConfig.parse_obj(obj)


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
                data_loader=tmp_file_path,
                model_loader=tmp_file_path,
                model_evaluator=tmp_file_path,
            )
            deserialized_yaml = yaml.safe_load(yaml.safe_dump(config.dict()))
            deserialized_config = ExperimentConfig(**deserialized_yaml)
            self.assertEqual(config.dict(), deserialized_config.dict())


class TestOptimizationResult(unittest.TestCase):
    @staticmethod
    def _new_optimization_result(
        optimized_model: Optional[OptimizedModelDescriptor] = None,
        latency_rate_improvement: Optional[float] = None,
        throughput_rate_improvement: Optional[float] = None,
        size_rate_improvement: Optional[float] = None,
    ) -> OptimizationResult:
        return OptimizationResult(
            hardware_info=HardwareInfo(
                cpu="",
                operating_system="",
                memory_gb=0,
                vm_size="",
                vm_provider=VMProvider.AZURE,
            ),
            optimized_model=optimized_model,
            original_model=OriginalModelDescriptor(
                name="",
                framework="",
                latency_seconds=0,
                throughput=0,
                size_mb=0,
            ),
            latency_improvement_rate=latency_rate_improvement,
            throughput_improvement_rate=throughput_rate_improvement,
            size_improvement_rate=size_rate_improvement,
        )

    def test_rates_validation__optimized_model_is_none(self):
        res = self._new_optimization_result()
        self.assertIsNotNone(res)

    def test_rates_validation__optimized_model_but_rates_are_none(self):
        optimized_model = OptimizedModelDescriptor(
            latency_seconds=0,
            throughput=0,
            size_mb=0,
            technique="",
            compiler="",
            metric_drop=0,
            model_path=Path(),
        )
        with self.assertRaises(ValidationError):
            self._new_optimization_result(
                optimized_model=optimized_model,
                latency_rate_improvement=None,
                throughput_rate_improvement=None,
                size_rate_improvement=None,
            )
        with self.assertRaises(ValidationError):
            self._new_optimization_result(
                optimized_model=optimized_model,
                latency_rate_improvement=1,
                throughput_rate_improvement=None,
                size_rate_improvement=None,
            )
        with self.assertRaises(ValidationError):
            self._new_optimization_result(
                optimized_model=optimized_model,
                latency_rate_improvement=1,
                throughput_rate_improvement=1,
                size_rate_improvement=None,
            )
