from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Optional, List

import ray
from ray import remote
from rich import print

from nebullvm.tools.base import DeviceType, Device
from nebullvm.tools.utils import gpu_is_available
from surfer import ModelLoader, DataLoader, ModelEvaluator
from surfer.common import schemas
from surfer.core.clusters import RayCluster, Accelerator
from surfer.log import logger
from surfer.optimization.models import OptimizeInferenceResult
from surfer.optimization.operations import (
    OptimizeInferenceOp,
)
from surfer.storage.clients import StorageClient
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
        accelerator: Accelerator,
        num_cpus: int = 1,
        num_gpus: int = 1,
    ):
        remote_decorator = remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            accelerator_type=accelerator.value,
        )
        self._optimize_inference = remote_decorator(self._optimize_inference)

    @staticmethod
    def _optimize_inference(config: RunConfig) -> schemas.OptimizationResult:
        def __get_optimize_fn() -> callable:
            op = OptimizeInferenceOp()
            if gpu_is_available():
                device = Device(DeviceType.GPU)
            else:
                device = Device(DeviceType.CPU)
            op.to(device)
            fn = op.execute
            if config.model_evaluator is not None:
                precision_fn = config.model_evaluator.get_precision_metric_fn
                fn = partial(fn, precision_metric_fn=precision_fn)
            return fn

        logger.info("starting inference optimization")
        optimize_fn = __get_optimize_fn()
        res: OptimizeInferenceResult = optimize_fn(
            model=config.model_loader.load_model(),
            input_data=config.data_loader.load_data(),
            metric_drop_ths=config.metric_drop_threshold,
            store_latencies=True,
            ignored_compilers=config.ignored_compilers,
        )
        print(res)
        return schemas.OptimizationResult( # TODO

        )

    def run(self, config: RunConfig) -> ray.ObjectRef:
        return self._optimize_inference.remote(config)


class RayOrchestrator:
    def __init__(
        self,
        cluster: RayCluster,
        storage_client: StorageClient,
    ):
        self.cluster = cluster
        self.storage_client = storage_client

    def run_experiment(self, experiment: str, config: RunConfig):
        # Setup tasks
        tasks = []
        for accelerator in self.cluster.get_available_accelerators():
            if accelerator in config.ignored_accelerators:
                continue
            tasks.append(InferenceOptimizationTask(accelerator))
        # Submit
        objs = []
        for t in tasks:
            objs.append(t.run(config))
        results = ray.get(objs)
        if len(results) == 0:
            print("No results")
        for r in results:
            print(r.json())
