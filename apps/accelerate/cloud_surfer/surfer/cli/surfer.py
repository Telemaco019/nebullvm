from pathlib import Path

import typer as typer

from surfer.cli.commands import cmd_init
from surfer.cli.commands import cmd_list_accelerators
from surfer.cli.experiments import app as experiment_app
from surfer.common import constants
from surfer.storage.models import StorageProvider

app = typer.Typer(no_args_is_help=True)
app.add_typer(
    experiment_app,
    no_args_is_help=True,
    name="experiment",
    help="Manage model optimization experiments",
)


@app.command(
    name="init",
    help="Init Cloud Surfer configuration",
)
def init(
    cluster_file: Path = typer.Argument(
        ...,
        metavar="cluster-file",
        help="Path to the Ray cluster YAML config file",
        exists=True,
        dir_okay=False,
    ),
    storage_provider: StorageProvider = typer.Option(
        ...,
        metavar="storage-provider",
        help="The cloud storage provider used "
        "for storing experiment data and optimized models",
    ),
    ray_address: str = typer.Option(
        constants.DEFAULT_RAY_ADDRESS,
        metavar="ray-address",
        help="Address of the head node of the Ray cluster",
    ),
):
    cmd_init.init(cluster_file, storage_provider, ray_address)


@app.command(
    name="list-accelerators",
    help="List the accelerators available in the configured cluster.",
)
def list_accelerators():
    cmd_list_accelerators.list_accelerators()


def main():
    app()
