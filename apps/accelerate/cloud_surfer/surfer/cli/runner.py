from pathlib import Path
from typing import Optional, List

import typer
import yaml
from typer import Typer

import surfer.log
from surfer.cli.commands import cmd_run as cmd
from surfer.common.schemas import SurferConfig
from surfer.computing.models import Accelerator
from surfer.core.orchestrators import RunConfig

app = Typer(no_args_is_help=True)


@app.callback()
def doc():
    """
    CLI for running surfer experiments. This CLI is not meant to be
    used by end users, but rather by Ray Jobs submitted by CloudSurfer.
    """


@app.command(name="run", help="test")
def run(
    results_dir: Path = typer.Option(
        ...,
        exists=False,  # the path refers to a remote storage
    ),
    data_loader_path: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    model_loader_path: Path = typer.Option(
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
    metric_drop_threshold: Optional[float] = typer.Option(
        None,
        min=0,
    ),
    ignored_accelerators: List[Accelerator] = typer.Option(
        [],
    ),
    ignored_compilers: List[str] = typer.Option(
        [],
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    surfer.log.setup_logger(debug)
    with open(surfer_config_path) as f:
        config_dict = yaml.safe_load(f.read())
        surfer_config = SurferConfig.parse_obj(config_dict)
    run_config = RunConfig(
        model_loader_path=model_loader_path,
        data_loader_path=data_loader_path,
        metric_drop_threshold=metric_drop_threshold,
        ignored_compilers=ignored_compilers,
        ignored_accelerators=ignored_accelerators,
        model_evaluator_path=model_evaluator_path,
    )
    cmd.run(surfer_config, run_config, results_dir)


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

    def with_surfer_config(self, path: Path) -> "RunCommandBuilder":
        self.__command += f" --surfer-config-path {path}"
        return self

    def with_results_dir(self, path: Path) -> "RunCommandBuilder":
        self.__command += f" --results-dir {path.as_posix()}"
        return self

    def with_metric_drop_threshold(self, t: float) -> "RunCommandBuilder":
        self.__command += f" --metric-drop-threshold {t}"
        return self

    def with_ignored_accelerators(
        self,
        accelerators: List[Accelerator],
    ) -> "RunCommandBuilder":
        for a in accelerators:
            self.__command += f" --ignored-accelerators {a.value}"
        return self

    def with_ignored_compilers(
        self,
        compilers: List[str],
    ) -> "RunCommandBuilder":
        for c in compilers:
            self.__command += f" --ignored-compilers {c}"
        return self

    def with_debug(self) -> "RunCommandBuilder":
        self.__command += f" --debug"
        return self

    def get_command(self) -> str:
        return self.__command


if __name__ == "__main__":
    app()
