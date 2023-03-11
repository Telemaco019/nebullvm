from pathlib import Path
from typing import Optional, List, Union

import yaml
from pydantic import BaseModel, FilePath, AnyUrl, validator
from pydantic.error_wrappers import ValidationError

from nebullvm.config import DEFAULT_METRIC_DROP_THS
from surfer import storage
from surfer.common import constants
from surfer.storage.models import StorageConfig, StorageProvider


class ModelDescriptor(BaseModel):
    class Config:
        frozen = True
        extra = "forbid"
        json_encoders = {
            Path: lambda v: v.resolve().as_posix(),
        }

    model_name: str
    framework: str
    latency: float
    metric_drop: float
    throughput: float
    model_size_mb: float
    technique: Optional[str] = None
    compiler: Optional[str] = None
    model_path: Optional[Path] = None


class HardwareInfo(BaseModel):
    class Config:
        frozen = True
        extra = "forbid"

    cpu: str
    operating_system: str
    memory_gb: int
    accelerator: str


class OptimizationResult(BaseModel):
    class Config:
        frozen = True
        extra = "forbid"
        json_encoders = {
            Path: lambda v: v.resolve().as_posix(),
        }

    hardware_info: HardwareInfo
    best_model: ModelDescriptor
    original_model: ModelDescriptor
    all_models: List[ModelDescriptor]


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
    storage: StorageConfig
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

    @classmethod
    def __parse_storage_config(cls, obj) -> StorageConfig:
        storage_config_dict = obj["storage"]
        provider = StorageProvider(storage_config_dict["provider"])

        if provider is StorageProvider.AZURE:
            if provider not in storage.enabled_providers:
                raise ValueError(
                    f"storage provider {provider.value} not installed. "
                    f'Please install "surfer\[azure]" to use it.'
                )
            from surfer.storage.providers.azure import AzureStorageConfig

            return AzureStorageConfig.parse_obj(storage_config_dict)

        if provider is StorageProvider.AWS:
            if provider not in storage.enabled_providers:
                raise ValueError(
                    f"storage provider {provider.value} not installed. "
                    f'Please install "surfer\[aws]" to use it.'
                )
            from surfer.storage.providers.aws import AWSStorageConfig

            return AWSStorageConfig.parse_obj(storage_config_dict)

        if provider is StorageProvider.GCP:
            if provider not in storage.enabled_providers:
                raise ValueError(
                    f"storage provider {provider.value} not installed. "
                    f'Please install "surfer\[gcp]" to use it.'
                )
            from surfer.storage.providers.gcp import GCPStorageConfig

            return GCPStorageConfig.parse_obj(storage_config_dict)

        raise ValidationError(f"storage provider {provider} is not supported")

    @classmethod
    def parse_obj(cls, obj) -> "SurferConfig":
        parsed: Union["SurferConfig", BaseModel] = super().parse_obj(obj)
        storage_config = cls.__parse_storage_config(obj)
        parsed.storage = storage_config
        return parsed

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
    data_loader: FilePath
    model_loader: FilePath
    model_evaluator: Optional[FilePath] = None
    additional_requirements: List[str] = []
    metric_drop_threshold: float = DEFAULT_METRIC_DROP_THS
    ignored_compilers: List[str] = []
    ignored_accelerators: List[str] = []

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
