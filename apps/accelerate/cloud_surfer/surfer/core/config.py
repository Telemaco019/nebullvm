import json
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel
from pydantic.class_validators import validator
from pydantic.types import FilePath

from surfer.core import constants
from surfer.storage.aws import AWSStorageConfig
from surfer.storage.azure import AzureStorageConfig
from surfer.storage.gcp import GCPStorageConfig


class SurferConfig(BaseModel):
    cluster_file: FilePath
    storage: Union[AzureStorageConfig, GCPStorageConfig, AWSStorageConfig]

    class Config:
        extra = "forbid"
        frozen = True

    @validator("cluster_file")
    def validate_cluster_file(cls, v):
        with open(v) as f:
            try:
                _ = yaml.safe_load(f.read())
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"{v} is not a valid YAML: {e}")
        return v


class SurferConfigManager:
    def __init__(
            self,
            base_path: Path = constants.SURFER_CONFIG_BASE_DIR_PATH,
            config_file_name=constants.SURFER_CONFIG_FILE_NAME,
    ):
        self.config_file_path = base_path / config_file_name

    def config_exists(self) -> bool:
        return self.config_file_path.exists()

    def save_config(self, config: SurferConfig):
        # Create config file
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file_path, "w") as f:
            config_dict = json.loads(config.json())
            f.write(yaml.dump(config_dict))

    def load_config(self) -> Optional[SurferConfig]:
        if not self.config_exists():
            return None
        try:
            with open(self.config_file_path) as f:
                config_dict = yaml.safe_load(f.read())
                return SurferConfig.parse_obj(config_dict)
        except Exception as e:
            raise Exception(f"Error parsing CloudSurfer config at {self.config_file_path}: {e}")
