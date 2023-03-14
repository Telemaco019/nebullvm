from pathlib import Path
from typing import Optional

from surfer.common import schemas
from surfer.computing.models import VMProvider
from surfer.optimization.models import OptimizedModel, OriginalModel
from surfer.utilities.nebullvm_utils import HardwareSetup


class ModelDescriptor:
    @staticmethod
    def from_optimized_model(
        m: OptimizedModel,
        model_path: Optional[Path] = None,
    ) -> schemas.OptimizedModelDescriptor:
        return schemas.OptimizedModelDescriptor(
            model_id=m.model_id,
            latency=m.latency,
            throughput=m.throughput,
            metric_drop=m.metric_drop,
            technique=m.technique,
            model_size_mb=m.model_size_mb,
            compiler=m.compiler,
            model_path=model_path,
        )

    @staticmethod
    def from_original_model(
        m: OriginalModel,
    ) -> schemas.OriginalModelDescriptor:
        return schemas.OriginalModelDescriptor(
            model_id=m.model_id,
            latency=m.latency,
            throughput=m.throughput,
            model_size_mb=m.model_info.model_size_mb,
            model_name=m.model_info.model_name,
            framework=m.model_info.framework.value,
        )


class HardwareSetupConverter:
    @staticmethod
    def to_hw_info_schema(
        h: HardwareSetup,
        vm_size: str,
        vm_provider: VMProvider,
    ) -> schemas.HardwareInfo:
        return schemas.HardwareInfo(
            cpu=h.cpu,
            operating_system=h.operating_system,
            memory_gb=h.memory_gb,
            accelerator=h.gpu,
            vm_size=vm_size,
            vm_provider=vm_provider.value,
        )
