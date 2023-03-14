import unittest
from pathlib import Path
from unittest.mock import MagicMock

from nebullvm.tools.base import DeepLearningFramework
from surfer.computing.models import VMProvider
from surfer.optimization import converters
from surfer.optimization.models import OptimizedModel, OriginalModel, ModelInfo
from surfer.utilities.nebullvm_utils import HardwareSetup


class TestModelDescriptor(unittest.TestCase):
    def test_from_optimized_model__path_is_none(self):
        m = OptimizedModel(
            inference_learner=None,
            latency=0.0,
            throughput=0.0,
            metric_drop=0.0,
            technique="",
            compiler="",
            model_size_mb=0.0,
        )
        res = converters.ModelDescriptor.from_optimized_model(m)
        self.assertIsNotNone(res)

    def test_from_optimized_model__path_is_not_none(self):
        path = Path("path")
        m = OptimizedModel(
            inference_learner=None,
            latency=0.0,
            throughput=0.0,
            metric_drop=0.0,
            technique="",
            compiler="",
            model_size_mb=0.0,
        )
        res = converters.ModelDescriptor.from_optimized_model(
            m,
            model_path=path,
        )
        self.assertIsNotNone(res)
        self.assertEqual(path, res.model_path)

    def test_from_original_model(self):
        m = OriginalModel(
            model_info=ModelInfo(
                model_name="",
                model_size_mb=0.0,
                framework=DeepLearningFramework.PYTORCH,
            ),
            latency=0.0,
            throughput=0.0,
            model=MagicMock(),
        )
        res = converters.ModelDescriptor.from_original_model(m)
        self.assertIsNotNone(res)


class TestHardwareSetupConverter(unittest.TestCase):
    def test_to_hw_info_schema(self):
        h = HardwareSetup(
            cpu="",
            operating_system="",
            memory_gb=0,
            gpu="",
        )
        res = converters.HardwareSetupConverter.to_hw_info_schema(
            h, vm_size="test", vm_provider=VMProvider.AZURE
        )
        self.assertIsNotNone(res)
