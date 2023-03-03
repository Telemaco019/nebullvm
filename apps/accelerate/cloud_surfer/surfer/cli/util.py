import logging
from datetime import datetime

from surfer.core import constants
from surfer.log import logger


def configure_debug_mode(debug: bool):
    if debug is True:
        logger.level = logging.DEBUG
        logger.debug("debug mode enabled")


def format_datetime_ui(date: datetime) -> str:
    return date.strftime(constants.DISPLAYED_DATETIME_FORMAT)
