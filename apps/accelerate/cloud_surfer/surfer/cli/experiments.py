import typer
from rich import print
from rich.table import Table

from surfer.core import services
from surfer.core import util
from surfer.core.config import SurferConfig, SurferConfigManager
from surfer.log import logger
from surfer.storage.clients import StorageClient

app = typer.Typer()

config_manager = SurferConfigManager()


def _must_get_config() -> SurferConfig:
    config = config_manager.load_config()
    if config is None:
        logger.error("Cloud Surfer is not initialized. Please run `surfer init` first.")
        raise typer.Exit(1)
    if not config.cluster_file.exists():
        logger.error(
            f"Surfer configuration refers to a non-existent cluster file: {config.cluster_file}."
            "\nPlease run `surfer init` to re-initialize the configuration."
        )
        raise typer.Exit(1)
    return config


@app.command(
    name="list",
    help="List all the experiments"
)
def list_experiments():
    # Init services
    config = _must_get_config()
    storage_client = StorageClient.from_config(config.storage)
    experiment_service = services.ExperimentService(storage_client=storage_client)

    # List experiments
    experiments = experiment_service.list()

    # Render
    table = Table()
    table.add_column("Experiment", header_style="cyan")
    table.add_column("Status", header_style="cyan")
    table.add_column("Created at", header_style="cyan")
    for e in experiments:
        table.add_row(e.name, e.status, util.format_datetime(e.created_at))
    print(table)


@app.command(
    name="submit",
    help="Submit a new model optimization experiment",
)
def submit_experiment():
    pass


@app.command(
    name="stop",
    help="Stop a running experiment",
)
def stop_experiment():
    pass


@app.command(
    name="describe",
    help="Show the details of an experiment",
)
def describe_experiment():
    pass


@app.command(
    name="delete",
    help="Delete an experiment. If the experiment is running, it will be stopped and deleted."
)
def delete_experiment():
    pass
