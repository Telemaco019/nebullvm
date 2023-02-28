from pathlib import Path

import typer

from surfer.core.config import SurferConfigManager
from surfer.log import logger

config_manager = SurferConfigManager()


def must_get_cluster_file() -> Path:
    config = config_manager.load_config()
    if config is None:
        logger.error("Cloud Surfer is not initialized. Please run `surfer init` first.")
        raise typer.Exit(1)
    if not config.cluster_file.exists():
        logger.error(
            f"Surfer configuration refers to a non-existent cluster file: {config.cluster_file}."
            "\nPlease run `surfer init` to re-initialize the configuration."
        )
        raise typer.Exit(1)
    return config.cluster_file
