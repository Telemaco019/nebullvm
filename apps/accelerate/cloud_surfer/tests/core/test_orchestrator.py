import unittest
from unittest.mock import patch, MagicMock

from surfer.core.orchestrators import SpeedsterOptimizer, RunConfig
from tests import _get_assets_path


class TestSpeedsterOptimizer(unittest.TestCase):
    @patch("surfer.core.orchestrators.optimize_model")
    def test_run__no_model_evaluator(self, mocked_optimize_model):
        config = RunConfig(
            model_loader_path=_get_assets_path() / "model_loaders.py",
            data_loader_path=_get_assets_path() / "data_loaders.py",
            ignored_compilers=[],
            ignored_accelerators=[],
            metric_drop_threshold=0.1,
        )
        optimizer = SpeedsterOptimizer(config)
        optimizer.run()
        mocked_optimize_model.assert_called_once()
        self.assertIsNone(
            mocked_optimize_model.call_args.kwargs.get(
                "precision_metric_fn",
                None,
            )
        )

    @patch("surfer.core.orchestrators.optimize_model")
    def test_run__model_evaluator(self, mocked_optimize_model):
        config = RunConfig(
            model_loader_path=_get_assets_path() / "model_loaders.py",
            data_loader_path=_get_assets_path() / "data_loaders.py",
            model_evaluator_path=_get_assets_path() / "model_evaluators.py",
            ignored_compilers=[],
            ignored_accelerators=[],
            metric_drop_threshold=0.1,
        )
        optimizer = SpeedsterOptimizer(config)
        optimizer.run()
        mocked_optimize_model.assert_called_once()
        self.assertIsInstance(
            mocked_optimize_model.call_args.kwargs["precision_metric_fn"],
            MagicMock,
        )
