from pathlib import Path

import typer

from surfer.core.config import SurferConfigManager
from surfer.log import logger

config_manager = SurferConfigManager()


def must_get_cluster_config() -> Path:
    config = config_manager.load_config()
    if config is None:
        logger.error("Cloud Surfer is not initialized. Please run `surfer init` first.")
        raise typer.Exit(1)
    return config.cluster_config
