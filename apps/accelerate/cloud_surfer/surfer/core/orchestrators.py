from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Optional, List

import ray
from ray import remote
from speedster.api.functions import optimize_model

from surfer import ModelLoader, DataLoader, ModelEvaluator
from surfer.core.clusters import RayCluster, Accelerator
from surfer.log import logger
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
        logger.debug("loading model loader", self.model_loader_path)
        loader = ClassLoader[ModelLoader](ModelLoader)
        return loader.load_from_module(self.model_loader_path)()

    @cached_property
    def data_loader(self) -> DataLoader:
        logger.debug("loading data loader", self.data_loader)
        loader = ClassLoader[DataLoader](DataLoader)
        return loader.load_from_module(self.data_loader_path)()

    @cached_property
    def model_evaluator(self) -> Optional[ModelEvaluator]:
        if self.model_evaluator is None:
            return None
        logger.debug("loading model evaluator", self.model_evaluator)
        loader = ClassLoader[ModelEvaluator](ModelEvaluator)
        return loader.load_from_module(self.model_evaluator_path)()


def speedster_optimize(config: RunConfig):
    def __get_optimize_model_fn() -> callable:
        fn = partial(
            optimize_model,
            model=config.model_loader.load_model(),
            input_data=config.data_loader.load_data(),
            metric_drop_ths=config.metric_drop_threshold,
            ignored_compilers=config.ignored_compilers,
        )
        if config.model_evaluator is not None:
            precision_fn = config.model_evaluator.get_precision_metric_fn
            fn = partial(fn, precision_metric_fn=precision_fn())
        return fn

    optimize_fn = __get_optimize_model_fn()
    optimize_fn()


class RayOrchestrator:
    def __init__(
        self,
        cluster: RayCluster,
        storage_client: StorageClient,
    ):
        self.cluster = cluster
        self.storage_client = storage_client

    def run_experiment(self, experiment: str, config: RunConfig):
        tasks = []
        # Add actors on accelerators
        for accelerator in self.cluster.get_available_accelerators():
            if accelerator in config.ignored_accelerators:
                continue
            remote_accelerator = remote(
                num_cpus=1,
                num_gpus=1,
                accelerator_type=accelerator.value,
            )
            tasks.append(remote_accelerator(speedster_optimize))
        # Add actors on CPU
        task = remote(num_cpus=1)(speedster_optimize)
        tasks.append(task)
        # Submit runs
        objs = []
        for t in tasks:
            objs.append(t.remote(config))
        results = ray.get(objs)
        print(results)
