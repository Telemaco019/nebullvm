import sys
from pathlib import Path
from typing import Optional

import ray
import typer
from typer import Typer

import surfer.log
from surfer.common.schemas import SurferConfig

app = Typer(no_args_is_help=True)


@app.callback()
def doc():
    """
    CLI for running surfer experiments. This CLI is not meant to be
    used by end users, but rather by Ray Jobs submitted by CloudSurfer.
    """


@ray.remote(num_gpus=1)
def read_path(p: Path):
    print("Python version is ", sys.version)
    # Run nvidia-smi and print output
    print(ray.get_gpu_ids())
    print(f"Path is {p.as_posix()}")
    with open(p) as f:
        print("content is ", f.read())


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
    surfer_config_path: Path = typer.Option(
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
    surfer_config = SurferConfig.parse_file(surfer_config_path)
    res = read_path.remote(surfer_config_path)
    ray.get(res)


class RunCommandBuilder:
    def __init__(self, interpreter="python3"):
        from surfer.cli import runner  # noqa W605

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

    def with_surfer_config(self, path: Path) -> "RunCommandBuilder":
        self.__command += f" --surfer-config-path {path}"
        return self

    def with_debug(self) -> "RunCommandBuilder":
        self.__command += f" --debug"
        return self

    def get_command(self) -> str:
        return self.__command


if __name__ == "__main__":
    app()
