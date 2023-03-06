import asyncio
from pathlib import Path

import typer
import yaml
from pydantic.error_wrappers import ValidationError
from rich import print
from rich import progress
from rich.rule import Rule
from rich.table import Table

from surfer.cli import util
from surfer.core import services
from surfer.core.exceptions import NotFoundError, InternalError
from surfer.core.models import SubmitExperimentRequest
from surfer.core.schemas import SurferConfig, ExperimentConfig
from surfer.core.services import SurferConfigManager, ExperimentService
from surfer.log import logger

app = typer.Typer()

config_manager = SurferConfigManager()


def _must_load_config() -> SurferConfig:
    config = config_manager.load_config()
    if config is None:
        logger.error(
            "Cloud Surfer is not initialized. "
            "Please run `surfer init` first."
        )
        raise typer.Exit(1)
    return config


def _new_experiment_service() -> ExperimentService:
    config = _must_load_config()
    return services.Factory.new_experiment_service(config)


async def _list_experiments():
    # Init services
    config = _must_load_config()
    experiment_service = services.Factory.new_experiment_service(config)

    # List experiments
    experiments = await experiment_service.list()

    # Render
    table = Table()
    table.add_column("Experiment", header_style="cyan")
    table.add_column("Status", header_style="cyan")
    table.add_column("Created at", header_style="cyan")
    for e in experiments:
        table.add_row(
            e.name,
            str(e.status),
            util.format_datetime_ui(e.created_at),
        )
    print(table)


@app.command(
    name="list",
    help="List all the experiments"
)
def list_experiments(
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    util.configure_debug_mode(debug)
    asyncio.run(_list_experiments())


def _load_experiments_config(path: Path) -> ExperimentConfig:
    try:
        with progress.open(
            path,
            "r",
            description=f"Loading experiment config...",
        ) as f:
            content = yaml.safe_load(f.read())
            return ExperimentConfig.parse_obj(content)
    except ValidationError as e:
        logger.error(f"Error parsing experiment config", e)
        raise typer.Exit(1)


async def _submit_experiment(name: str, config_path: Path):
    config = _load_experiments_config(config_path)
    # Init services
    experiment_service = _new_experiment_service()
    # Submit experiment
    req = SubmitExperimentRequest(
        config=config,
        name=name,
    )
    try:
        await experiment_service.submit(req)
    except (InternalError, ValueError) as e:
        logger.error(f"Failed to submit experiment: {e}")
        raise typer.Exit(1)


@app.command(
    name="submit",
    help="Submit a new model optimization experiment",
)
def submit_experiment(
    experiment_config: Path = typer.Argument(
        ...,
        metavar="experiment-config",
        help="YAML file containing the experiment configuration",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    name: str = typer.Option(
        ...,
        metavar="name",
        help="The name of the experiment. Must be unique.",
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    util.configure_debug_mode(debug)
    asyncio.run(_submit_experiment(name, experiment_config))


async def _stop_experiment(name: str):
    # Init services
    experiment_service = _new_experiment_service()
    # Stop experiment
    try:
        await experiment_service.stop(name)
    except NotFoundError:
        logger.error(f"Experiment not found")
        raise typer.Exit(1)
    except (InternalError, ValueError) as e:
        logger.error(f"Failed to stop experiment: {e}")
        raise typer.Exit(1)
    logger.info(f"Experiment stopped")


@app.command(
    name="stop",
    help="Stop a running experiment",
)
def stop_experiment(
    name: str = typer.Argument(
        ...,
        metavar="name",
        help="The name of the experiment to stop",
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    util.configure_debug_mode(debug)
    typer.confirm(f"Are you sure you want to stop experiment {name}?",
                  abort=True)
    asyncio.run(_stop_experiment(name))


async def _describe_experiment(name: str):
    # Init services
    experiment_service = _new_experiment_service()
    # Fetch experiment
    try:
        experiment = await experiment_service.get(name)
    except InternalError as e:
        logger.error(f"Failed to fetch experiment: {e}")
        raise typer.Exit(1)
    if experiment is None:
        logger.error("Experiment not found")
        raise typer.Exit(1)
    # Render summary
    print(Rule("Summary"))
    print("[bold]Experiment name[/bold]: {}".format(experiment.name))
    print("[bold]Created at[/bold]: {}".format(
        util.format_datetime_ui(experiment.created_at))
    )
    print("[bold]Status[/bold]: {}".format(experiment.status))
    # List jobs
    print(Rule("Jobs"))
    jobs_table = Table(box=None)
    jobs_table.add_column("ID", header_style="cyan")
    jobs_table.add_column("Status", header_style="cyan")
    jobs_table.add_column("Details", header_style="cyan")
    for j in experiment.jobs:
        jobs_table.add_row(j.job_id, j.status, j.additional_info)
    if len(experiment.jobs) == 0:
        print("No jobs")
    else:
        print(jobs_table)
    # Show results
    print(Rule("Results"))


@app.command(
    name="describe",
    help="Show the details of an experiment",
)
def describe_experiment(
    name: str = typer.Argument(
        ...,
        metavar="name",
        help="The name of the experiment to describe",
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    util.configure_debug_mode(debug)
    asyncio.run(_describe_experiment(name))


async def _delete_experiment(name: str):
    # Init services
    experiment_service = _new_experiment_service()
    # Delete experiment
    try:
        await experiment_service.delete(name)
    except NotFoundError:
        logger.error(f"Experiment not found")
        raise typer.Exit(1)
    except InternalError as e:
        logger.error(f"Failed to delete experiment: {e}")
        raise typer.Exit(1)
    logger.info(f"Experiment deleted")


@app.command(
    name="delete",
    help="Delete an experiment. "
         "If the experiment is running, it will be stopped and deleted."
)
def delete_experiment(
    name: str = typer.Argument(
        ...,
        metavar="name",
        help="The name of the experiment to delete",
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    util.configure_debug_mode(debug)
    typer.confirm(f"Are you sure you want to delete experiment {name}?",
                  abort=True)
    asyncio.run(_delete_experiment(name))
