import unittest
from pathlib import Path
from unittest.mock import MagicMock

from nebullvm.tools.base import DeepLearningFramework
from surfer.computing.models import VMProvider
from surfer.optimization import converters
from surfer.optimization.models import (
    OptimizedModel,
    OriginalModel,
    OptimizeInferenceResult,
)
from surfer.utilities.nebullvm_utils import HardwareSetup


class TestModelDescriptor(unittest.TestCase):
    def test_from_optimized_model__path_is_not_none(self):
        path = Path("path")
        m = OptimizedModel(
            inference_learner=MagicMock(),
            latency_seconds=0.0,
            throughput=0.0,
            metric_drop=0.0,
            technique="",
            compiler="",
            size_mb=0.0,
        )
        res = converters.ModelDescriptor.from_optimized_model(
            m,
            model_path=path,
        )
        self.assertIsNotNone(res)
        self.assertEqual(path, res.model_path)

    def test_from_original_model(self):
        m = OriginalModel(
            name="",
            size_mb=0.0,
            framework=DeepLearningFramework.PYTORCH,
            latency_seconds=0.0,
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


class TestInferenceResultConverter(unittest.TestCase):
    def test_to_optimization_result__no_optimal_model(self):
        o = OptimizeInferenceResult(
            original_model=OriginalModel(
                name="",
                size_mb=0.0,
                framework=DeepLearningFramework.PYTORCH,
                latency_seconds=0.0,
                throughput=0.0,
                model=MagicMock(),
            ),
            optimized_model=None,
            hardware_setup=HardwareSetup(
                cpu="",
                operating_system="",
                memory_gb=0,
            ),
        )
        res = converters.InferenceResultConverter.to_optimization_result(
            res=o,
            vm_size="test",
            vm_provider=VMProvider.AZURE,
        )
        self.assertIsNotNone(res)

    def test_to_optimization_result__with_optimal_model(self):
        o = OptimizeInferenceResult(
            original_model=OriginalModel(
                name="",
                size_mb=0.0,
                framework=DeepLearningFramework.PYTORCH,
                latency_seconds=0.0,
                throughput=0.0,
                model=MagicMock(),
            ),
            optimized_model=OptimizedModel(
                inference_learner=MagicMock(),
                latency_seconds=0.0,
                throughput=0.0,
                metric_drop=0.0,
                technique="",
                compiler="",
                size_mb=0.0,
            ),
            hardware_setup=HardwareSetup(
                cpu="",
                operating_system="",
                memory_gb=0,
            ),
        )
        res = converters.InferenceResultConverter.to_optimization_result(
            res=o,
            vm_size="test",
            vm_provider=VMProvider.AZURE,
            optimized_model_path=Path("/tmp/invalid"),
        )
        self.assertIsNotNone(res)
