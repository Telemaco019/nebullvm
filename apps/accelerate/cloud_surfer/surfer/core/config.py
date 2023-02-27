from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml

from surfer.core import constants


@dataclass
class SurferConfig:
    cluster_config: Path

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

    def create_config(self, cluster_config: Path):
        # Check cluster config path
        if not cluster_config.exists():
            raise FileNotFoundError(f"Ray cluster YAML config file not found: {cluster_config}")

        # Validate cluster config
        with open(cluster_config) as f:
            try:
                _ = yaml.safe_load(f.read())
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"{cluster_config} is not a valid YAML: {e}")

        # Create config file
        self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
        config = SurferConfig(cluster_config=cluster_config.absolute())
        with open(self.config_file_path, "w") as f:
            f.write(yaml.dump(config.dict()))

    def load_config(self) -> Optional[SurferConfig]:
        if not self.config_exists():
            return None
        with open(self.config_file_path) as f:
            config_dict = yaml.safe_load(f.read())
        try:
            return SurferConfig(**config_dict)
        except Exception as e:
            raise Exception(f"Error while parsing config file: {e}")
