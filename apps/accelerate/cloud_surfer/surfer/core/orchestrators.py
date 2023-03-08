from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Optional, List

from speedster.api.functions import optimize_model

from surfer import ModelLoader, DataLoader, ModelEvaluator
from surfer.common.schemas import SpeedsterResult
from surfer.core.clusters import RayCluster
from surfer.log import logger
from surfer.storage.clients import StorageClient
from surfer.utilities.python_utils import ClassLoader


@dataclass
class RunConfig:
    model_loader: Path
    data_loader: Path
    metric_drop_threshold: float
    ignored_compilers: List[str]
    model_evaluator: Optional[Path] = None


class SpeedsterOptimizer:
    def __init__(self, config: RunConfig):
        self.config = config

    @cached_property
    def model_loader(self) -> ModelLoader:
        logger.debug("loading model loader", self.config.model_loader)
        loader = ClassLoader[ModelLoader](ModelLoader)
        return loader.load_from_module(self.config.model_loader)()

    @cached_property
    def data_loader(self) -> DataLoader:
        logger.debug("loading data loader", self.config.data_loader)
        loader = ClassLoader[DataLoader](DataLoader)
        return loader.load_from_module(self.config.data_loader)()

    @cached_property
    def model_evaluator(self) -> Optional[ModelEvaluator]:
        if self.config.model_evaluator is None:
            return None
        logger.debug("loading model evaluator", self.config.model_evaluator)
        loader = ClassLoader[ModelEvaluator](ModelEvaluator)
        return loader.load_from_module(self.config.model_evaluator)()

    def __get_optimize_model_fn(self) -> callable:
        optimize_fn = partial(
            optimize_model,
            model=self.model_loader.load_model(),
            input_data=self.data_loader.load_data(),
            metric_drop_ths=self.config.metric_drop_threshold,
            ignored_compilers=self.config.ignored_compilers,
        )
        if self.model_evaluator is not None:
            precision_fn = self.model_evaluator.get_precision_metric_fn
            optimize_fn = partial(
                optimize_fn,
                precision_metric_fn=precision_fn(),
            )
        return optimize_fn

    def run(self) -> SpeedsterResult:
        optimize_fn = self.__get_optimize_model_fn()
        optimize_fn()
        return None


class OrchestrationService:
    def __init__(
        self,
        cluster: RayCluster,
        storage_client: StorageClient,
    ):
        self.cluster = cluster
        self.storage_client = storage_client

    def run_experiment(self, experiment: str, config: RunConfig):
        pass
