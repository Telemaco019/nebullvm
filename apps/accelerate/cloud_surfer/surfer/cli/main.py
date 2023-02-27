from pathlib import Path

import typer as typer
from rich.prompt import Confirm

from surfer.cli.experiments import app as experiment_app
from surfer.core.config import SurferConfigManager
from surfer.log import logger

app = typer.Typer(no_args_is_help=True)
app.add_typer(
    experiment_app,
    no_args_is_help=True,
    name="experiments",
    help="Manage model optimization experiments",
)


@app.command(
    name="init",
    help="Init Cloud Surfer configuration",
)
def init(
        cluster_config: Path = typer.Argument(
            ...,
            metavar="cluster-config-file",
            help="Path to the Ray cluster YAML config file",
            exists=True,
            dir_okay=False,
        )
):
    config_manager = SurferConfigManager()
    write_config = True
    if config_manager.config_exists():
        write_config = Confirm.ask(
            "Configuration already exists. Do you want to overwrite it?",
            default=False,
            show_choices=True,
        )
    if write_config:
        config_manager.create_config(cluster_config=cluster_config)
        config = config_manager.load_config()
        logger.info("Cloud Surfer configuration initialized", config)


def main():
    app()
