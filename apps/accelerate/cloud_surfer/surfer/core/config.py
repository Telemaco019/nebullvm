from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml

from surfer.core import constants


@dataclass
class SurferConfig:
    cluster_file: Path
    bucket_signed_url: str

    def __validate_cluster_file(self):
        # Check cluster config path
        if not self.cluster_file.exists():
            raise FileNotFoundError(f"Ray cluster YAML config file not found: {self.cluster_file}")
        if not self.cluster_file.is_file():
            raise FileNotFoundError(f"{self.cluster_file} is not a file")
        # Validate cluster config
        with open(self.cluster_file) as f:
            try:
                _ = yaml.safe_load(f.read())
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"{self.cluster_file} is not a valid YAML: {e}")

    def __post_init__(self):
        self.__validate_cluster_file()

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


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
            f.write(yaml.dump(config.dict()))

    def load_config(self) -> Optional[SurferConfig]:
        if not self.config_exists():
            return None
        try:
            with open(self.config_file_path) as f:
                config_dict = yaml.safe_load(f.read())
                return SurferConfig(cluster_file=Path(config_dict["cluster_file"]))
        except Exception as e:
            raise Exception(f"Error parsing CloudSurfer config at {self.config_file_path}: {e}")
