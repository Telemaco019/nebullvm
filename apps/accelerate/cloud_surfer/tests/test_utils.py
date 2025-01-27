from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import yaml

from surfer.common.schemas import (
    ExperimentConfig,
    OptimizedModelDescriptor,
    OptimizationResult,
    OriginalModelDescriptor,
)
from surfer.computing.schemas import VMPricing, HardwareInfo, VMInfo
from surfer.storage.models import StorageConfig, StorageProvider


@contextmanager
def tmp_experiment_config_file() -> Path:
    with TemporaryDirectory() as d:
        tmp_file = Path(d, "config.yaml")
        with open(tmp_file, "w") as f:
            config = ExperimentConfig(
                description="test",
                data_loader=Path(f.name),
                model_loader=Path(f.name),
            )
            f.write(yaml.dump(config.dict()))
        yield tmp_file


class MockedStorageConfig(StorageConfig):
    provider = StorageProvider.AZURE


def new_optimization_result(
    optimized_model: Optional[OptimizedModelDescriptor] = None,
    latency_rate_improvement: Optional[float] = None,
    throughput_rate_improvement: Optional[float] = None,
    size_rate_improvement: Optional[float] = None,
) -> OptimizationResult:
    return OptimizationResult(
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
        optimized_model=optimized_model,
        original_model=OriginalModelDescriptor(
            name="",
            framework="",
            latency_seconds=0,
            throughput=0,
            size_mb=0,
        ),
        latency_improvement_rate=latency_rate_improvement,
        throughput_improvement_rate=throughput_rate_improvement,
        size_improvement_rate=size_rate_improvement,
    )
