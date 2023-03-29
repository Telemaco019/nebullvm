import logging
import sys

from loguru import logger
from rich.console import Console


class _Console:
    def __init__(self):
        self._error_console = Console(stderr=True, style="red")
        self.__console = Console()

    def print(self, *objs: any, **kwargs):
        self.__console.print(*objs, **kwargs)

    def warn(self, *objs: any, **kwargs):
        self.__console.print(
            "\[warning]",
            *objs,
            style="yellow",
            **kwargs,
        )

    def error(self, *objs: any, **kwargs):
        self._error_console.print(*objs, **kwargs)


console = _Console()
level = logging.INFO


def setup_logger(debug: bool = False):
    global level
    if debug is True:
        level = logging.DEBUG
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=None,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | <level>{message}</level>"
        ),
        level=level,
    )
