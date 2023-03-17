from typing import Optional

import typer
from pydantic.error_wrappers import ValidationError

from surfer.common import constants
from surfer.common.schemas import SurferConfig
from surfer.core.config import SurferConfigManager
from surfer.log import logger

config_manager = SurferConfigManager()


def must_load_config() -> SurferConfig:
    try:
        config = config_manager.load_config()
    except ValidationError as e:
        logger.error(e)
        raise typer.Exit(1)

    if config is None:
        logger.error(
            "Cloud Surfer is not initialized. Please run `surfer init` first.",
        )
        raise typer.Exit(1)

    return config


def format_float(f: float, precision=2) -> str:
    return f"{f:.{precision}f}"


def format_rate(rate: Optional[float]) -> str:
    if rate is None:
        return constants.NOT_AVAILABLE_MSG
    return "{:.1f}x".format(rate)
