import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from surfer import ModelLoader, DataLoader, ModelEvaluator
from surfer.core.clusters import Accelerator
from surfer.core.orchestrators import RunConfig, RayOrchestrator
from tests import _get_assets_path


class TestRunConfig(unittest.TestCase):
    def test_model_loader(self):
        base_path = _get_assets_path()
        data_loader = Path(base_path, "data_loaders.py")
        model_loader = Path(base_path, "model_loaders.py")
        config = RunConfig(
            model_loader_path=model_loader,
            data_loader_path=data_loader,
            ignored_compilers=[],
            metric_drop_threshold=0,
            ignored_accelerators=[],
        )
        self.assertIsNotNone(config.model_loader)
        self.assertIsInstance(config.model_loader, ModelLoader)

    def test_data_loader(self):
        base_path = _get_assets_path()
        data_loader = Path(base_path, "data_loaders.py")
        model_loader = Path(base_path, "model_loaders.py")
        config = RunConfig(
            model_loader_path=model_loader,
            data_loader_path=data_loader,
            ignored_compilers=[],
            metric_drop_threshold=0,
            ignored_accelerators=[],
        )
        self.assertIsNotNone(config.data_loader)
        self.assertIsInstance(config.data_loader, DataLoader)

    def test_model_evaluator(self):
        base_path = _get_assets_path()
        data_loader = Path(base_path, "data_loaders.py")
        model_loader = Path(base_path, "model_loaders.py")
        model_evaluator = Path(base_path, "model_evaluators.py")
        config = RunConfig(
            model_loader_path=model_loader,
            data_loader_path=data_loader,
            model_evaluator_path=model_evaluator,
            ignored_compilers=[],
            metric_drop_threshold=0,
            ignored_accelerators=[],
        )
        self.assertIsNotNone(config.model_evaluator)
        self.assertIsInstance(config.model_evaluator, ModelEvaluator)


@patch("surfer.core.orchestrators.ray")
class TestRayOrchestrator(unittest.TestCase):
    @patch("surfer.core.orchestrators.InferenceOptimizationTask")
    def test_run_experiment__no_available_accelerators(self, mocked_task, *_):
        mocked_ray_cluster = MagicMock()
        mocked_ray_cluster.get_available_accelerators.return_value = []
        orchestrator = RayOrchestrator(
            cluster=mocked_ray_cluster,
            storage_client=MagicMock(),
            surfer_config=MagicMock(),
        )
        orchestrator.run_experiment(MagicMock(), MagicMock())
        mocked_task.run.assert_not_called()

    @patch("surfer.core.orchestrators.InferenceOptimizationTask")
    def test_run_experiment__at_least_one_task_per_accelerator(
        self,
        mocked_task,
        *_,
    ):
        mocked_ray_cluster = MagicMock()
        accelerators = [
            Accelerator.NVIDIA_TESLA_K80,
            Accelerator.NVIDIA_TESLA_V100,
        ]
        mocked_ray_cluster.get_available_accelerators.return_value = accelerators
        orchestrator = RayOrchestrator(
            cluster=mocked_ray_cluster,
            storage_client=MagicMock(),
            surfer_config=MagicMock(),
        )
        orchestrator.run_experiment(MagicMock(), MagicMock())
        self.assertGreaterEqual(mocked_task.call_count, len(accelerators))

    @patch("surfer.core.orchestrators.InferenceOptimizationTask")
    def test_run_experiment__ignored_accelerators(
        self,
        mocked_task,
        *_,
    ):
        mocked_ray_cluster = MagicMock()
        accelerators = [
            Accelerator.NVIDIA_TESLA_K80,
        ]
        mocked_ray_cluster.get_available_accelerators.return_value = accelerators
        orchestrator = RayOrchestrator(
            cluster=mocked_ray_cluster,
            storage_client=MagicMock(),
            surfer_config=MagicMock(),
        )
        orchestrator.run_experiment(
            MagicMock(),
            RunConfig(
                model_loader_path=MagicMock(),
                data_loader_path=MagicMock(),
                ignored_accelerators=[Accelerator.NVIDIA_TESLA_K80],
                ignored_compilers=[],
                metric_drop_threshold=0,
            ),
        )
        mocked_task.assert_not_called()
