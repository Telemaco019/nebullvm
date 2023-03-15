import asyncio
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, List

import ray
from ray import remote
from rich import print

from nebullvm.tools.base import DeviceType, Device
from nebullvm.tools.utils import gpu_is_available
from surfer import ModelLoader, DataLoader, ModelEvaluator
from surfer.common import schemas, constants
from surfer.common.schemas import SurferConfig
from surfer.computing.clusters import ClusterNode
from surfer.computing.clusters import RayCluster, Accelerator
from surfer.computing.models import VMProvider
from surfer.log import logger
from surfer.optimization import converters
from surfer.optimization.models import OptimizeInferenceResult, OptimizedModel
from surfer.optimization.operations import (
    OptimizeInferenceOp,
)
from surfer.storage.clients import StorageClient
from surfer.storage.models import StorageConfig
from surfer.utilities.python_utils import ClassLoader


@dataclass
class RunConfig:
    model_loader_path: Path
    data_loader_path: Path
    metric_drop_threshold: float
    ignored_compilers: List[str]
    ignored_accelerators: List[Accelerator]
    model_evaluator_path: Optional[Path] = None

    @cached_property
    def model_loader(self) -> ModelLoader:
        logger.info("loading model loader", self.model_loader_path)
        loader = ClassLoader[ModelLoader](ModelLoader)
        return loader.load_from_module(self.model_loader_path)()

    @cached_property
    def data_loader(self) -> DataLoader:
        logger.info("loading data loader", self.data_loader_path)
        loader = ClassLoader[DataLoader](DataLoader)
        return loader.load_from_module(self.data_loader_path)()

    @cached_property
    def model_evaluator(self) -> Optional[ModelEvaluator]:
        if self.model_evaluator_path is None:
            return None
        logger.info("loading model evaluator", self.model_evaluator_path)
        loader = ClassLoader[ModelEvaluator](ModelEvaluator)
        return loader.load_from_module(self.model_evaluator_path)()


class InferenceOptimizationTask:
    def __init__(
        self,
        node: ClusterNode,
        num_cpus: int = 1,
        num_gpus: int = 1,
    ):
        remote_decorator = remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            accelerator_type=node.accelerator.value,
        )
        self.node = node
        self._run = remote_decorator(self._run)

    @staticmethod
    def _upload_model(
        storage_config: StorageConfig,
        results_dir: Path,
        model: OptimizedModel,
    ) -> Path:
        Path(model.model_id).mkdir(exist_ok=True)
        with TemporaryDirectory() as tmp:
            dir_path = Path(tmp)
            model.inference_learner.save(dir_path)
            client = StorageClient.from_config(storage_config)
            dest_path = Path(
                results_dir,
                constants.INFERENCE_LEARNERS_DIR_NAME,
                model.model_id,
            )
            asyncio.run(client.upload(source=dir_path, dest=dest_path))
            return dest_path

    @staticmethod
    def _optimize_inference(config: RunConfig) -> OptimizeInferenceResult:
        def __get_device() -> Device:
            if gpu_is_available():
                return Device(DeviceType.GPU)
            return Device(DeviceType.CPU)

        # Setup
        device = __get_device()
        metric_fn = None
        if config.model_evaluator is not None:
            metric_fn = config.model_evaluator.get_precision_metric_fn()
        # Optimize
        op = OptimizeInferenceOp()
        op.to(device)
        res = op.execute(
            model=config.model_loader.load_model(),
            input_data=config.data_loader.load_data(),
            metric_drop_ths=config.metric_drop_threshold,
            store_latencies=True,
            metric=metric_fn,
            ignored_compilers=config.ignored_compilers,
        )
        return res

    @staticmethod
    def _run(
        storage_config: StorageConfig,
        results_dir: Path,
        run_config: RunConfig,
        vm_size: str,
        vm_provider: VMProvider,
    ) -> schemas.OptimizationResult:
        # Run inference optimization
        logger.info("starting inference optimization")
        res = InferenceOptimizationTask._optimize_inference(run_config)
        # Extract best model (if present) and upload it to storage
        optimized_model_path: Optional[Path] = None
        if res.optimized_model is not None:
            optimized_model_path = InferenceOptimizationTask._upload_model(
                storage_config,
                results_dir,
                res.optimized_model,
            )
        else:
            logger.warning("optimization didn't produce any best model")
        # Return result
        return converters.InferenceResultConverter.to_optimization_result(
            res,
            vm_size,
            vm_provider,
            optimized_model_path,
        )

    def run(
        self,
        storage_config: StorageConfig,
        results_dir: Path,
        run_config: RunConfig,
        vm_size: str,
        vm_provider: VMProvider,
    ) -> ray.ObjectRef:
        return self._run.remote(
            storage_config,
            results_dir,
            run_config,
            vm_size,
            vm_provider,
        )


class RayOrchestrator:
    def __init__(
        self,
        cluster: RayCluster,
        storage_client: StorageClient,
        surfer_config: SurferConfig,
    ):
        self.cluster = cluster
        self.surfer_config = surfer_config
        self.storage_client = storage_client

    def run_experiment(
        self,
        results_dir: Path,
        config: RunConfig,
    ):
        # Setup tasks
        tasks = []
        for node in self.cluster.get_nodes():
            if node.accelerator in config.ignored_accelerators:
                continue
            tasks.append(InferenceOptimizationTask(node))
        # Submit
        objs = []
        for t in tasks:
            o = t.run(
                storage_config=self.surfer_config.storage,
                results_dir=results_dir,
                run_config=config,
                vm_size=t.node.vm_size,
                vm_provider=self.cluster.provider,
            )
            objs.append(o)
        results = ray.get(objs)
        if len(results) == 0:
            print("No results")
        for r in results:
            print(r.json())
