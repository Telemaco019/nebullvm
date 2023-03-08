from pathlib import Path

import typer
import yaml
from pydantic.error_wrappers import ValidationError
from rich import progress
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

import surfer.core.experiments
import surfer.utilities.datetime_utils
from surfer.cli.commands.common import must_load_config
from surfer.common.exceptions import NotFoundError, InternalError
from surfer.common.schemas import ExperimentConfig
from surfer.core.experiments import SubmitExperimentRequest
from surfer.log import logger


def _new_experiment_service() -> surfer.core.experiments.ExperimentService:
    config = must_load_config()
    return surfer.core.experiments.new_experiment_service(config)


async def list_experiments():
    # Init services
    config = must_load_config()
    experiment_service = surfer.core.experiments.new_experiment_service(config)

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
            surfer.utilities.datetime_utils.format_datetime_ui(e.created_at),
        )
    print(table)


def _load_experiments_config(path: Path) -> ExperimentConfig:
    try:
        with progress.open(
            path,
            "r",
            description="Loading experiment config...",
        ) as f:
            content = yaml.safe_load(f.read())
            return ExperimentConfig.parse_obj(content)
    except ValidationError as e:
        logger.error("Error parsing experiment config", e)
        raise typer.Exit(1)


async def submit_experiment(name: str, config_path: Path):
    config = _load_experiments_config(config_path)
    # Init services
    experiment_service = _new_experiment_service()
    # Submit experiment
    req = SubmitExperimentRequest(
        config=config,
        name=name,
    )
    try:
        logger.info("Submitting experiment...")
        await experiment_service.submit(req)
        logger.info("Experiment submitted successfully :tada:")
        logger.info("\nYou can check the status of the experiment with:")
        print(Panel(f"> [green]surfer experiment describe {req.name}[/green]"))
    except (InternalError, ValueError) as e:
        logger.error(f"Failed to submit experiment: {e}")
        raise typer.Exit(1)


async def stop_experiment(name: str):
    # Init services
    experiment_service = _new_experiment_service()
    # Stop experiment
    try:
        await experiment_service.stop(name)
    except NotFoundError:
        logger.error("Experiment not found")
        raise typer.Exit(1)
    except (InternalError, ValueError) as e:
        logger.error(f"Failed to stop experiment: {e}")
        raise typer.Exit(1)
    logger.info("Experiment stopped")


async def describe_experiment(name: str):
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
    print(
        "[bold]Created at[/bold]: {}".format(
            surfer.utilities.datetime_utils.format_datetime_ui(experiment.created_at)
        )
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
    if experiment.result is None:
        print("No results available")


async def delete_experiment(name: str):
    # Init services
    experiment_service = _new_experiment_service()
    # Delete experiment
    try:
        await experiment_service.delete(name)
    except NotFoundError:
        logger.error("Experiment not found")
        raise typer.Exit(1)
    except InternalError as e:
        logger.error(f"Failed to delete experiment: {e}")
        raise typer.Exit(1)
    logger.info("Experiment deleted")
