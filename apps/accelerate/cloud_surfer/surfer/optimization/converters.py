from pathlib import Path
from typing import Optional

from surfer.common import schemas
from surfer.computing.models import VMProvider
from surfer.computing.schemas import HardwareInfo, VMInfo
from surfer.optimization.models import (
    OptimizedModel,
    OriginalModel,
    OptimizeInferenceResult,
)
from surfer.utilities.nebullvm_utils import HardwareSetup


class ModelDescriptor:
    @staticmethod
    def from_optimized_model(
        m: OptimizedModel,
        model_path: Path,
    ) -> schemas.OptimizedModelDescriptor:
        return schemas.OptimizedModelDescriptor(
            latency_seconds=m.latency_seconds,
            throughput=m.throughput,
            metric_drop=m.metric_drop,
            technique=m.technique,
            size_mb=m.size_mb,
            compiler=m.compiler,
            model_path=model_path,
        )

    @staticmethod
    def from_original_model(
        m: OriginalModel,
    ) -> schemas.OriginalModelDescriptor:
        return schemas.OriginalModelDescriptor(
            latency_seconds=m.latency_seconds,
            throughput=m.throughput,
            size_mb=m.size_mb,
            name=m.name,
            framework=m.framework.value,
        )


class HardwareSetupConverter:
    @staticmethod
    def to_hw_info_schema(
        h: HardwareSetup,
    ) -> HardwareInfo:
        return HardwareInfo(
            cpu=h.cpu,
            operating_system=h.operating_system,
            memory_gb=h.memory_gb,
            accelerator=h.gpu,
        )


class InferenceResultConverter:
    @staticmethod
    def to_optimization_result(
        res: OptimizeInferenceResult,
        vm_sku: str,
        vm_provider: VMProvider,
        optimized_model_path: Optional[Path] = None,
    ) -> schemas.OptimizationResult:
        # Convert original model
        original_model_desc = ModelDescriptor.from_original_model(
            res.original_model,
        )
        # Convert hw setup
        hw_info = HardwareSetupConverter.to_hw_info_schema(
            res.hardware_setup,
        )
        vm_info = VMInfo(
            hardware_info=hw_info,
            sku=vm_sku,
            provider=vm_provider.value,
        )
        # Extract best model
        if res.optimized_model is not None:
            optimized_model_desc = ModelDescriptor.from_optimized_model(
                res.optimized_model,
                optimized_model_path,
            )
        else:
            return schemas.OptimizationResult(
                vm_info=vm_info,
                optimized_model=None,
                original_model=original_model_desc,
            )
        # Return result
        return schemas.OptimizationResult(
            vm_info=vm_info,
            optimized_model=optimized_model_desc,
            original_model=original_model_desc,
            latency_improvement_rate=res.latency_improvement_rate,
            throughput_improvement_rate=res.throughput_improvement_rate,
            size_improvement_rate=res.size_improvement_rate,
        )
