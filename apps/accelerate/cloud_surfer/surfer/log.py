import logging

from rich.console import Console


class Logger:
    """
    Wrapper of rich.Console that exposes methods that show output
    using different styles according to the respective log level.
    """

    def __init__(self, log_level=logging.INFO):
        self.level = log_level
        self.__console = Console()

    def info(self, *objs: any, **kwargs):
        if self.level <= logging.INFO:
            self.__console.print(*objs, **kwargs)

    def warn(self, *objs: any, **kwargs):
        if self.level <= logging.WARNING:
            self.__console.print("\[warning]", *objs, style="yellow", **kwargs)

    def debug(self, *objs: any, **kwargs):
        if self.level <= logging.DEBUG:
            self.__console.print("\[debug]", *objs, style="cyan", **kwargs)  # noqa

    def error(self, *objs: any, **kwargs):
        self.__console.print(*objs, style="red", **kwargs)


logger = Logger()
