import os
import platform
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cpuinfo
import psutil

from nebullvm.operations.conversions.converters import (
    PytorchConverter,
    TensorflowConverter,
    ONNXConverter,
    Converter,
)
from nebullvm.optional_modules import torch
from nebullvm.optional_modules.diffusers import diffusers
from nebullvm.optional_modules.utils import (
    torch_is_available,
    tensorflow_is_available,
)
from nebullvm.tools.base import Device, DeepLearningFramework, DeviceType
from nebullvm.tools.diffusers import DiffusionUNetWrapper, \
    is_diffusion_model_pipe
from nebullvm.tools.pytorch import torch_get_device_name
from nebullvm.tools.tf import tensorflow_get_gpu_name


# TODO: move this module to Nebullvm SDK


@dataclass
class HardwareSetup:
    cpu: str
    operating_system: str
    memory_gb: int
    gpu: Optional[str] = None


def get_model_size_mb(model: Any) -> float:
    if isinstance(model, str):
        size = os.stat(model).st_size
    elif isinstance(model, Path):
        size = os.path.getsize(model.as_posix())
    elif isinstance(model, torch.Module):
        size = sum(p.nelement() * p.element_size() for p in model.parameters())
    else:
        # we assume it is a tf_model
        # assuming full precision 32 bit
        size = model.count_params() * 4
    return round(size * 1e-6, 2)


def get_model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    if isinstance(model, Path):
        return model.as_posix()
    return model.__class__.__name__


def generate_model_id(model: Any) -> str:
    model_name = get_model_name(model)
    return f"{str(uuid.uuid4())}_{hash(model_name)}"


def get_hw_setup(device: Device) -> HardwareSetup:
    return HardwareSetup(
        cpu=cpuinfo.get_cpu_info()["brand_raw"],
        operating_system=platform.system(),
        memory_gb=round(psutil.virtual_memory().total * 1e-9, 2),
        gpu=get_gpu_name() if device.type is DeviceType.GPU else None,
    )


def get_gpu_name() -> str:
    if torch_is_available():
        name = torch_get_device_name()
    elif tensorflow_is_available():
        name = tensorflow_get_gpu_name()
    else:
        name = "Unknown"

    return name


def get_conversion_op(framework: DeepLearningFramework) -> Converter:
    if framework == DeepLearningFramework.PYTORCH:
        conversion_op = PytorchConverter()
    elif framework == DeepLearningFramework.TENSORFLOW:
        conversion_op = TensorflowConverter()
    else:
        conversion_op = ONNXConverter()

    return conversion_op


def get_throughput(latency: float, batch_size: int) -> float:
    return (1 / latency) * batch_size


def is_diffusion_model(model):
    from diffusers import UNet2DConditionModel

    if is_diffusion_model_pipe(model):
        return True
    if isinstance(model, (UNet2DConditionModel, DiffusionUNetWrapper)):
        return True
    if hasattr(model, "model"):
        return isinstance(model.model, diffusers.models.UNet2DConditionModel)
    return False
