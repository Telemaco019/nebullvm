import asyncio
import json
import uuid
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, List

import ray
from loguru import logger
from ray import remote

from nebullvm.tools.base import DeviceType, Device
from nebullvm.tools.utils import gpu_is_available
from surfer import ModelLoader, DataLoader, ModelEvaluator
from surfer.common import schemas, constants
from surfer.common.exceptions import InternalError
from surfer.common.schemas import SurferConfig
from surfer.computing.clusters import ClusterNode
from surfer.computing.clusters import RayCluster, Accelerator
from surfer.computing.models import VMProvider
from surfer.computing.schemas import VMPricing
from surfer.computing.services import PricingService
from surfer.optimization import converters
from surfer.optimization.models import OptimizeInferenceResult, OptimizedModel
from surfer.optimization.operations import (
    OptimizeInferenceOp,
)
from surfer.storage.clients import StorageClient
from surfer.storage.models import StorageConfig
from surfer.utilities import nebullvm_utils
from surfer.utilities.python_utils import ClassLoader


@dataclass
class RunConfig:
    model_loader_path: Path
    data_loader_path: Path
    ignored_compilers: List[str]
    ignored_accelerators: List[Accelerator]
    metric_drop_threshold: Optional[float] = None
    model_evaluator_path: Optional[Path] = None

    @cached_property
    def model_loader(self) -> ModelLoader:
        logger.info("loading model loader {}", self.model_loader_path)
        loader = ClassLoader[ModelLoader](ModelLoader)
        return loader.load_from_module(self.model_loader_path)()

    @cached_property
    def data_loader(self) -> DataLoader:
        logger.info("loading data loader {}", self.data_loader_path)
        loader = ClassLoader[DataLoader](DataLoader)
        return loader.load_from_module(self.data_loader_path)()

    @cached_property
    def model_evaluator(self) -> Optional[ModelEvaluator]:
        if self.model_evaluator_path is None:
            return None
        logger.info("loading model evaluator {}", self.model_evaluator_path)
        loader = ClassLoader[ModelEvaluator](ModelEvaluator)
        return loader.load_from_module(self.model_evaluator_path)()


class InferenceOptimizationTask:
    def __init__(
        self,
        node: ClusterNode,
        num_cpus: int = 1,
        num_gpus: int = 1,
    ):
        remote_decorator = self.__get_remote_decorator(
            node,
            num_cpus,
            num_gpus,
        )
        self.node = node
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._run = remote_decorator(self._run)

    @staticmethod
    def __get_remote_decorator(
        node: ClusterNode,
        num_cpus: int,
        num_gpus: int,
    ):
        if node.accelerator.is_tpu():
            return remote(
                num_cpus=0,
                num_gpus=0,
                resources={
                    node.accelerator.value: 1,
                },
            )
        return remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            accelerator_type=node.accelerator.value,
        )

    def __str__(self):
        return json.dumps(
            {
                "node": {
                    "vm_size": self.node.vm_size,
                    "accelerator": self.node.accelerator.value,
                },
                "num_cpus": self._num_cpus,
                "num_gpus": self._num_gpus,
            },
            indent=2,
        )

    @staticmethod
    def _upload_model(
        storage_config: StorageConfig,
        results_dir: Path,
        vm_size: str,
        model: OptimizedModel,
    ) -> Path:
        with TemporaryDirectory() as tmp:
            dir_path = Path(tmp, model.inference_learner.name)
            model.inference_learner.save(dir_path)
            client = StorageClient.from_config(storage_config)
            dest_path = Path(
                results_dir,
                constants.INFERENCE_LEARNERS_DIR_NAME,
                vm_size,
                str(uuid.uuid4()),
            )
            logger.info("uploading inference learner to {}".format(dest_path))
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
        node: ClusterNode,
        vm_provider: VMProvider,
    ) -> str:
        # Run inference optimization
        logger.info("starting inference optimization")
        inference_res = InferenceOptimizationTask._optimize_inference(
            run_config,
        )
        # Extract best model (if present) and upload it to storage
        optimized_model_path: Optional[Path] = None
        if inference_res.optimized_model is not None:
            logger.info("saving inference learner")
            optimized_model_path = InferenceOptimizationTask._upload_model(
                storage_config,
                results_dir,
                node.vm_size,
                inference_res.optimized_model,
            )
        else:
            logger.warning("optimization didn't produce any inference learner")
        # Prepare result
        result = converters.InferenceResultConverter.to_optimization_result(
            inference_res,
            node.vm_size,
            vm_provider,
            optimized_model_path,
        )
        # Get pricing info
        pricing_service = PricingService.from_provider(vm_provider)
        pricing: Optional[VMPricing] = None
        try:
            pricing = asyncio.run(
                pricing_service.get_vm_pricing(
                    vm_sku=node.vm_size,
                    region=node.region,
                )
            )
        except InternalError as e:
            logger.exception("failed to get pricing info", e)
        result.vm_info.pricing = pricing
        return result.json()

    def run(
        self,
        storage_config: StorageConfig,
        results_dir: Path,
        run_config: RunConfig,
        node: ClusterNode,
        vm_provider: VMProvider,
    ) -> ray.ObjectRef:
        return self._run.remote(
            storage_config,
            results_dir,
            run_config,
            node,
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

    def save_results(
        self,
        results_dir: Path,
        model_name: str,
        task_results: List[schemas.OptimizationResult],
    ):
        res = schemas.ExperimentResult(
            optimizations=task_results,
            model_name=model_name,
        )
        coro = self.storage_client.upload_content(
            content=res.json(),
            dest=results_dir / constants.EXPERIMENT_RESULT_FILE_NAME,
        )
        asyncio.run(coro)

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
        logger.info("submitting {} tasks", tasks)
        objs = []
        for t in tasks:
            logger.debug("submitting task {}", t)
            o = t.run(
                storage_config=self.surfer_config.storage,
                results_dir=results_dir,
                run_config=config,
                node=t.node,
                vm_provider=self.cluster.provider,
            )
            objs.append(o)
        logger.info("waiting for results...")
        results_json = ray.get(objs)
        logger.info("collected {} results", results_json)
        if len(results_json) == 0:
            logger.warning("optimization tasks produced no results")
            return
        # Deserialize results
        results = [schemas.OptimizationResult.parse_raw(r) for r in results_json]
        # Get original model info
        model = config.model_loader.load_model()
        model_name = nebullvm_utils.get_model_name(model)
        # Save results
        logger.info("saving results to directory {}", results_dir)
        self.save_results(results_dir, model_name, results)
