from pathlib import Path
from typing import Optional, List, Union

import yaml
from pydantic import BaseModel, FilePath, Field, AnyUrl, validator

from surfer.core import constants
from surfer.storage.aws import AWSStorageConfig
from surfer.storage.azure import AzureStorageConfig
from surfer.storage.gcp import GCPStorageConfig


class ExperimentConfig(BaseModel):
    description: Optional[str]
    data_loader_module: Path
    model_loader_module: Path
    model_evaluator_module: Optional[Path]
    additional_requirements: List[str] = []

    class Config:
        extra = "forbid"


class SurferConfig(BaseModel):
    """
    CloudSurfer configuration.

    Attributes
    ----------
    cluster_file: Path
        Path to the Ray cluster YAML file
    storage: Union[AzureStorageConfig, GCPStorageConfig, AWSStorageConfig]
        Storage configuration
    ray_address: str
        Address of the head node of the Ray cluster
    """
    cluster_file: FilePath = Field()
    storage: Union[AzureStorageConfig, GCPStorageConfig, AWSStorageConfig]
    ray_address: AnyUrl = constants.DEFAULT_RAY_ADDRESS

    class Config:
        extra = "forbid"
        frozen = True
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
