import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from surfer import ModelLoader, DataLoader, ModelEvaluator
from surfer.common import schemas
from surfer.computing.clusters import Accelerator, ClusterNode
from surfer.computing.schemas import VMInfo, HardwareInfo, VMPricing
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
        mocked_ray_cluster.get_nodes.return_value = []
        orchestrator = RayOrchestrator(
            cluster=mocked_ray_cluster,
            storage_client=MagicMock(),
            surfer_config=MagicMock(),
        )
        orchestrator.run_experiment(MagicMock(), MagicMock())
        mocked_task.run.assert_not_called()

    @patch("surfer.core.orchestrators.InferenceOptimizationTask")
    def test_run_experiment__at_least_one_task_per_node(
        self,
        mocked_task,
        *_,
    ):
        mocked_ray_cluster = MagicMock()
        nodes = [
            ClusterNode(
                accelerator=Accelerator.NVIDIA_TESLA_V100,
                vm_size="Standard_NC6",
            ),
            ClusterNode(
                accelerator=Accelerator.NVIDIA_TESLA_K80,
                vm_size="test",
            ),
        ]
        mocked_ray_cluster.get_nodes.return_value = nodes
        orchestrator = RayOrchestrator(
            cluster=mocked_ray_cluster,
            storage_client=MagicMock(),
            surfer_config=MagicMock(),
        )
        orchestrator.run_experiment(MagicMock(), MagicMock())
        self.assertGreaterEqual(mocked_task.call_count, len(nodes))

    @patch("surfer.core.orchestrators.InferenceOptimizationTask")
    def test_run_experiment__ignored_accelerators(
        self,
        mocked_task,
        *_,
    ):
        mocked_ray_cluster = MagicMock()
        v100_node = ClusterNode(
            vm_size="",
            accelerator=Accelerator.NVIDIA_TESLA_V100,
        )
        nodes = [
            v100_node,
            ClusterNode(
                vm_size="",
                accelerator=Accelerator.NVIDIA_TESLA_K80,
            ),
        ]
        mocked_ray_cluster.get_nodes.return_value = nodes
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
        mocked_task.assert_called_once()
        mocked_task.assert_called_with(v100_node)

    def test_save_results(self, *_):
        mocked_storage_client = AsyncMock()
        orchestrator = RayOrchestrator(
            cluster=MagicMock(),
            storage_client=mocked_storage_client,
            surfer_config=MagicMock(),
        )
        results = [
            schemas.OptimizationResult(
                vm_info=VMInfo(
                    sku="Standard_NC6",
                    provider="azure",
                    pricing=VMPricing(
                        currency="USD",
                        region="westus",
                        price_hr=1.1,
                        price_hr_spot=1.2,
                        price_hr_1yr=2.3,
                        price_hr_3yr=3.5,
                    ),
                    hardware_info=HardwareInfo(
                        cpu="",
                        operating_system="",
                        memory_gb=0,
                    ),
                ),
                original_model=schemas.OriginalModelDescriptor(
                    name="test",
                    framework="pytorch",
                    latency_seconds=0.0,
                    throughput=0.0,
                    size_mb=0.0,
                ),
            ),
            schemas.OptimizationResult(
                vm_info=VMInfo(
                    sku="Standard_NC6",
                    provider="azure",
                    pricing=VMPricing(
                        currency="USD",
                        region="westus",
                        price_hr=1.1,
                        price_hr_spot=1.2,
                        price_hr_1yr=2.3,
                        price_hr_3yr=3.5,
                    ),
                    hardware_info=HardwareInfo(
                        cpu="",
                        operating_system="",
                        memory_gb=0,
                    ),
                ),
                original_model=schemas.OriginalModelDescriptor(
                    name="test",
                    framework="pytorch",
                    latency_seconds=0.0,
                    throughput=0.0,
                    size_mb=0.0,
                ),
            ),
        ]
        orchestrator.save_results(Path(), "test", results)
        mocked_storage_client.upload_content.assert_called_once()
