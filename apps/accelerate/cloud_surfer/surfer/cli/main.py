from pathlib import Path

import typer as typer
from rich.prompt import Confirm

from surfer.cli.experiments import app as experiment_app
from surfer.core import storage
from surfer.core.config import SurferConfigManager, SurferConfig
from surfer.log import logger

app = typer.Typer(no_args_is_help=True)
app.add_typer(
    experiment_app,
    no_args_is_help=True,
    name="experiments",
    help="Manage model optimization experiments",
)


def _validate_signed_url(url: str):
    try:
        storage.SignedURL(url)
    except Exception as e:
        raise typer.BadParameter(str(e))


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
        bucket_signed_url: str = typer.Option(
            ...,
            help="The signed URL to the cloud bucket where the experiment data and optimized models will be stored",
            prompt=True,
            callback=_validate_signed_url,
        ),
):
    config_manager = SurferConfigManager()
    write_config = True
    if config_manager.config_exists():
        write_config = Confirm.ask(
            "Found existing CloudSurfer configuration. Do you want to overwrite it?",
            default=False,
            show_choices=True,
        )
    if write_config:
        config = SurferConfig(
            cluster_file=cluster_file,
            bucket_signed_url=bucket_signed_url,
        )
        config_manager.save_config(config)
        logger.info("Cloud Surfer configuration initialized", config_manager.load_config())


def main():
    app()
