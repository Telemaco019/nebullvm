from pathlib import Path
from typing import Optional, List, Union

import yaml
from pydantic import BaseModel, FilePath, AnyUrl, validator

from surfer.common import constants
from surfer.storage.providers.aws import AWSStorageConfig
from surfer.storage.providers.azure import AzureStorageConfig
from surfer.storage.providers.gcp import GCPStorageConfig


class OptimizationResult(BaseModel):
    technique: str
    compiler: str
    latency_seconds: float
    model_size_mb: float
    throughput: float
    memory_footprint_mb: Optional[float]
    model: Optional[Path]


class HardwareInfo(BaseModel):
    cpu: str
    operating_system: str
    memory: str
    accelerator: str


class SpeedsterResult(BaseModel):
    optimizations: List[OptimizationResult]
    best_model_latency: OptimizationResult
    best_model_memory: OptimizationResult
    hardware_info: HardwareInfo
    speedster_version: str


class SurferConfig(BaseModel):
    """
    Cloud Surfer configuration.

    Attributes
    ----------
    cluster_file: Path
        Path to the Ray cluster YAML file
    storage: Union[AzureStorageConfig, GCPStorageConfig, AWSStorageConfig]
        Storage configuration
    ray_address: str
        Address of the head node of the Ray cluster
    """

    cluster_file: FilePath
    storage: Union[AzureStorageConfig, GCPStorageConfig, AWSStorageConfig]
    ray_address: AnyUrl = constants.DEFAULT_RAY_ADDRESS

    class Config:
        extra = "forbid"
        json_encoders = {
            Path: lambda v: v.resolve().as_posix(),
        }

    @validator("cluster_file")
    def validate_cluster_file(cls, v):
        with open(v) as f:
            try:
                _ = yaml.safe_load(f.read())
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"{v} is not a valid YAML: {e}")
        return v

    def dict(self, *args, **kwargs):
        res = super().dict(*args, **kwargs)
        for k, v in res.items():
            if isinstance(v, Path):
                res[k] = v.resolve().as_posix()
            if isinstance(v, AnyUrl):
                res[k] = str(v)
        return res


class ExperimentConfig(BaseModel):
    description: Optional[str]
    data_loader_module: FilePath
    model_loader_module: FilePath
    model_evaluator_module: Optional[FilePath]
    additional_requirements: List[str] = []

    class Config:
        extra = "forbid"
        frozen = True
        json_encoders = {
            Path: lambda v: v.resolve().as_posix(),
        }

    def dict(self, *args, **kwargs):
        res = super().dict(*args, **kwargs)
        for k, v in res.items():
            if isinstance(v, Path):
                res[k] = v.resolve().as_posix()
        return res


class ExperimentResult(BaseModel):
    pass