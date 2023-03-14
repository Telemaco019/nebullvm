import unittest
from unittest.mock import MagicMock

from surfer.optimization.models import (
    OriginalModel,
    OptimizedModel,
    OptimizeInferenceResult,
)


class TestOriginalModel(unittest.TestCase):
    def test_get_id(self):
        model = OriginalModel(
            model=MagicMock(),
            model_info=MagicMock(),
            latency=0.0,
            throughput=0.0,
        )
        ids = [model.model_id for _ in range(10)]
        self.assertTrue(len(ids[0]) > 0)
        self.assertTrue(all([i == ids[0] for i in ids]))


class TestOptimizedModel(unittest.TestCase):
    def test_get_id(self):
        model = OptimizedModel(
            inference_learner=MagicMock(),
            latency=0.0,
            throughput=0.0,
            metric_drop=0.0,
            technique="",
            compiler="",
            size_mb=0.0,
        )
        ids = [model.model_id for _ in range(10)]
        self.assertTrue(all([i is not None for i in ids]))
        self.assertTrue(len(ids[0]) > 0)
        self.assertTrue(all([i == ids[0] for i in ids]))


class TestOptimizeInferenceResult(unittest.TestCase):
    def test_lowest_latency_model__no_models(self):
        result = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_models=[],
        )
        self.assertIsNone(result.lowest_latency_model)

    def test_lowest_latency_model__no_inference_learners(self):
        result = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_models=[
                OptimizedModel(
                    inference_learner=None,
                    latency=0.0,
                    throughput=0.0,
                    metric_drop=0.0,
                    technique="",
                    compiler="",
                    size_mb=0.0,
                )
            ],
        )
        self.assertIsNone(result.lowest_latency_model)
    def test_lowest_latency_model(self):
        result = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_models=[
                OptimizedModel(
                    inference_learner=MagicMock(),
                    latency=0.0,
                    throughput=0.0,
                    metric_drop=0.0,
                    technique="",
                    compiler="",
                    size_mb=0.0,
                ),
                OptimizedModel(
                    inference_learner=MagicMock(),
                    latency=5.0,
                    throughput=0.0,
                    metric_drop=0.0,
                    technique="",
                    compiler="",
                    size_mb=0.0,
                )
            ],
        )
        self.assertIsNotNone(result.lowest_latency_model)
        self.assertEqual(result.lowest_latency_model.latency, 0.0) 

    def test_inference_learners__no_models(self):
        result = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_models=[],
        )
        self.assertEqual(len(result.inference_learners), 0)

    def test_inference_learners__no_inference_learners(self):
        result = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_models=[
                OptimizedModel(
                    inference_learner=None,
                    latency=0.0,
                    throughput=0.0,
                    metric_drop=0.0,
                    technique="",
                    compiler="",
                    size_mb=0.0,
                )
            ],
        )
        self.assertEqual(len(result.inference_learners), 0)

    def test_inference_learners__multiple_inference_learners(self):
        result = OptimizeInferenceResult(
            original_model=MagicMock(),
            hardware_setup=MagicMock(),
            optimized_models=[
                OptimizedModel(
                    inference_learner=MagicMock(),
                    latency=0.0,
                    throughput=0.0,
                    metric_drop=0.0,
                    technique="",
                    compiler="",
                    size_mb=0.0,
                )
            ],
        )
        self.assertEqual(len(result.inference_learners), 1)
