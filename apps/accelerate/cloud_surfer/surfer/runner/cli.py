from pathlib import Path
from typing import Optional

import typer
from typer import Typer

import surfer.log
from surfer import runner

app = Typer(no_args_is_help=True)


@app.callback()
def doc():
    """
    CLI for running surfer experiments. This CLI is not meant to be
    used by end users, but rather by Ray Jobs submitted by CloudSurfer.
    """


@app.command(name="run", help="test")
def run(
    experiment_name: str = typer.Option(
        ...,
    ),
    data_loader_path: str = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_loader_path: str = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_evaluator_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    storage_config_path: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    surfer.log.configure_debug_mode(debug)


class RunCommandBuilder:
    def __init__(self, interpreter="python3"):
        self.__command = f"{interpreter} -m {runner.__name__} {run.__name__}"

    def with_model_loader(
        self,
        model_loader_path: Path,
    ) -> "RunCommandBuilder":
        self.__command += f" --model-loader-path {model_loader_path}"
        return self

    def with_data_loader(
        self,
        data_loader_path: Path,
    ) -> "RunCommandBuilder":
        self.__command += f" --data-loader-path {data_loader_path}"
        return self

    def with_model_evaluator(
        self,
        model_evaluator_path: Path,
    ) -> "RunCommandBuilder":
        self.__command += f" --model-evaluator-path {model_evaluator_path}"
        return self

    def with_experiment_name(
        self,
        experiment_name: str,
    ) -> "RunCommandBuilder":
        self.__command += f" --experiment-name {experiment_name}"
        return self

    def with_storage_config(self, path: Path) -> "RunCommandBuilder":
        self.__command += f" --storage-config-path {path}"
        return self

    def with_debug(self) -> "RunCommandBuilder":
        self.__command += f" --debug"
        return self

    def get_command(self) -> str:
        return self.__command


def main():
    app()
