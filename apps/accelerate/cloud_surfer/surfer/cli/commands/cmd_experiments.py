import random
from pathlib import Path

import typer
import yaml
from pydantic.error_wrappers import ValidationError
from rich import box
from rich import print
from rich import progress
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

import surfer.core.experiments
import surfer.utilities.datetime_utils
from surfer.cli.commands.common import must_load_config, format_float
from surfer.common import schemas
from surfer.common.exceptions import NotFoundError, InternalError
from surfer.common.schemas import ExperimentConfig
from surfer.core.experiments import ExperimentDetails
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


def _render_optimization_result(res: schemas.OptimizationResult):
    print(Rule(style="bold white"))

    # Hardware info
    hw_table = Table(
        box=box.SIMPLE,
        header_style="bold white",
        title_style="italic yellow",
        title=res.hardware_info.vm_size,
        expand=True,
    )
    hw_table.add_column("Cloud Provider")
    hw_table.add_column("VM Size")
    hw_table.add_column("Accelerator")
    hw_table.add_column("CPU")
    hw_table.add_column("Memory (GB)")
    hw_table.add_column("Operating System")
    hw_table.add_row(
        res.hardware_info.vm_provider.value,
        res.hardware_info.vm_size,
        res.hardware_info.accelerator,
        res.hardware_info.cpu,
        str(res.hardware_info.memory_gb),
        res.hardware_info.operating_system,
    )
    print(hw_table)

    # Models
    table = Table(
        box=box.ROUNDED,
        header_style="bold cyan",
        expand=True,
        show_footer=True,
    )
    table.add_column("", footer="Improvement")
    table.add_column("Backend")
    table.add_column("Technique")
    table.add_column("Latency (ms)", footer="3.4x")
    table.add_column("Throughput (batch/sec)", footer="1x")
    table.add_column("Size", footer="0%")
    table.add_row(
        "Original",
        res.original_model.framework,
        "[italic]None[/italic]",
        format_float(res.original_model.latency),
        format_float(res.original_model.throughput),
        format_float(res.original_model.size_mb),
    )
    table.add_row(
        "Optimized",
        res.optimized_model.compiler,
        res.optimized_model.technique,
        format_float(res.optimized_model.latency),
        format_float(res.optimized_model.throughput),
        format_float(res.optimized_model.size_mb),
        style="bold green",
    )
    print(table)


def _render_experiment_results(experiment: ExperimentDetails):
    print(Rule("Results"))
    if experiment.result is None:
        print("No results available")
        return
    for optimization in experiment.result.optimizations:
        _render_optimization_result(optimization)


def _render_experiment_summary(experiment: ExperimentDetails):
    def __format_float(original: float, optimized: float) -> str:
        return "Original: {}\nOptimized: {}".format(
            format_float(original), format_float(optimized)
        )

    print(Rule("Summary"))
    print("[bold]Experiment name[/bold]: {}".format(experiment.name))
    print(
        "[bold]Created at[/bold]: {}".format(
            surfer.utilities.datetime_utils.format_datetime_ui(experiment.created_at)
        )
    )
    print("[bold]Status[/bold]: {}".format(experiment.status))

    # Results summary
    if experiment.result is None:
        return
    results_summary_table = Table(
        header_style="bold cyan",
        expand=True,
        box=box.SIMPLE,
    )
    results_summary_table.add_column("")
    results_summary_table.add_column("Accelerator")
    results_summary_table.add_column("Latency (ms)")
    results_summary_table.add_column("Throughput (batch/sec)")
    results_summary_table.add_column("Cost ($/inference)")
    for o in experiment.result.optimizations:
        results_summary_table.add_row(
            o.hardware_info.vm_size,
            o.hardware_info.accelerator,
            __format_float(
                o.original_model.latency,
                o.optimized_model.latency,
            ),
            __format_float(
                o.original_model.throughput,
                o.optimized_model.throughput,
            ),
            __format_float(
                random.randint(1, 1000) / 1000,
                random.randint(1, 1000) / 1000,
            ),
        )
    print(results_summary_table)

    print(
        "[bold]Lowest latency[/bold]: [green]{}[/green] ({} ms)".format(
            experiment.result.optimizations[0].hardware_info.vm_size,
            format_float(experiment.result.optimizations[0].optimized_model.latency),
        )
    )
    print(
        "[bold]Lowest cost[/bold]: [green]{}[/green] ({} $/inference)".format(
            experiment.result.optimizations[0].hardware_info.vm_size,
            format_float(random.randint(1, 1000) / 1000),
        )
    )


def _render_experiment_jobs(experiment: ExperimentDetails):
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


async def describe_experiment(name: str):
    # Init services
    experiment_service = _new_experiment_service()
    # Fetch experiment
    try:
        experiment: ExperimentDetails = await experiment_service.get(name)
    except InternalError as e:
        logger.error(f"Failed to fetch experiment: {e}")
        raise typer.Exit(1)
    if experiment is None:
        logger.error("Experiment not found")
        raise typer.Exit(1)
    # Render
    _render_experiment_summary(experiment)
    _render_experiment_jobs(experiment)
    _render_experiment_results(experiment)


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
