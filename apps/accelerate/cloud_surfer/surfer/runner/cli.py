from pathlib import Path
from typing import Optional

import typer
from typer import Typer

from surfer.storage.models import StorageConfig

app = Typer()


@app.command()
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
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    storage_config: StorageConfig = typer.Option(
        ...,
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    pass


def main():
    app()
