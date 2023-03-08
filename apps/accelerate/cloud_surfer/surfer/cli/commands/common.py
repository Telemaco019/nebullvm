import typer

from surfer.common.schemas import SurferConfig
from surfer.core.config import SurferConfigManager
from surfer.log import logger

config_manager = SurferConfigManager()


def must_load_config() -> SurferConfig:
    config = config_manager.load_config()
    if config is None:
        logger.error(
            "Cloud Surfer is not initialized. " "Please run `surfer init` first."
        )
        raise typer.Exit(1)
    return config
