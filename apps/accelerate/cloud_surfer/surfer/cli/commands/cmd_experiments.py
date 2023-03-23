from pathlib import Path

import typer
import yaml
from pydantic.error_wrappers import ValidationError
from rich import box
from rich import print
from rich import progress
from rich.panel import Panel
from rich.progress import SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table

import surfer.core.experiments
import surfer.utilities.datetime_utils
from surfer.cli.commands.common import (
    must_load_config,
    format_float,
    format_rate,
)
from surfer.common import schemas, constants
from surfer.common.exceptions import NotFoundError, InternalError
from surfer.common.schemas import ExperimentConfig
from surfer.core.experiments import ExperimentDetails
from surfer.core.experiments import SubmitExperimentRequest
from surfer.log import logger


def _new_experiment_service() -> surfer.core.experiments.ExperimentService:
    config = must_load_config()
    return surfer.core.experiments.new_experiment_service(config)


async def list_experiments():
    columns = [TextColumn("Loading experiments..."), SpinnerColumn()]
    with progress.Progress(*columns, transient=True) as progress_bar:
        progress_bar.add_task("")

        # Init services
        config = must_load_config()
        experiment_service = surfer.core.experiments.new_experiment_service(config)

        # List experiments
        experiments = await experiment_service.list()
        if len(experiments) == 0:
            print("No experiments available")
            return

        # Render
        table = Table(box=box.SIMPLE)
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

    print("\nYou can view experiment details with:")
    print(Panel(f"> [green]surfer experiment describe <experiment>[/green]"))


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
        title=res.vm_info.sku,
        expand=True,
    )
    hw_table.add_column("Cloud Provider")
    hw_table.add_column("VM Size")
    hw_table.add_column("Accelerator")
    hw_table.add_column("CPU")
    hw_table.add_column("Memory (GB)")
    hw_table.add_column("Operating System")
    hw_table.add_row(
        res.vm_info.provider.value,
        res.vm_info.sku,
        res.vm_info.hardware_info.accelerator,
        res.vm_info.hardware_info.cpu,
        str(res.vm_info.hardware_info.memory_gb),
        res.vm_info.hardware_info.operating_system,
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
    table.add_column(
        "Latency (ms)",
        footer=format_rate(res.latency_improvement_rate),
    )
    table.add_column(
        "Throughput (batch/sec)",
        footer=format_rate(res.throughput_improvement_rate),
    )
    table.add_column(
        "Size (MB)",
        footer=format_rate(res.size_improvement_rate),
    )
    table.add_row(
        "Original",
        res.original_model.framework,
        "[italic]None[/italic]",
        format_float(res.original_model.latency_ms),
        format_float(res.original_model.throughput),
        format_float(res.original_model.size_mb),
    )
    if res.optimized_model is None:
        table.add_row(
            "Optimized",
            constants.NOT_AVAILABLE_MSG,
            constants.NOT_AVAILABLE_MSG,
            constants.NOT_AVAILABLE_MSG,
            constants.NOT_AVAILABLE_MSG,
            constants.NOT_AVAILABLE_MSG,
            style="bold red",
        )
        print(table)
    else:
        table.add_row(
            "Optimized",
            res.optimized_model.compiler,
            res.optimized_model.technique,
            format_float(res.optimized_model.latency_ms),
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
    def __format_float(original: float, optimized: float, **kwargs) -> str:
        return "Original: {}\nOptimized: {}".format(
            format_float(original, **kwargs), format_float(optimized, **kwargs)
        )

    def __format_str(original: str, optimized: str) -> str:
        return "Original: {}\nOptimized: {}".format(original, optimized)

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
    print(
        "[bold]Model[/bold]: [yellow]{}[/yellow]".format(
            experiment.result.model_name,
        )
    )
    results_summary_table = Table(
        header_style="bold cyan",
        expand=True,
        box=box.SIMPLE,
    )
    results_summary_table.add_column("")
    results_summary_table.add_column("Accelerator")
    results_summary_table.add_column("Region")
    results_summary_table.add_column("$/hr")
    results_summary_table.add_column("Latency (ms)")
    results_summary_table.add_column("Throughput (batch/sec)")
    results_summary_table.add_column("$/inference")
    for o in experiment.result.optimizations:
        if o.optimized_model is None:
            continue
        results_summary_table.add_row(
            o.vm_info.sku,
            o.vm_info.hardware_info.accelerator,
            o.vm_info.pricing.region,
            format_float(o.vm_info.pricing.price_hr),
            __format_float(
                o.original_model.latency_ms,
                o.optimized_model.latency_ms,
            ),
            __format_float(
                o.original_model.throughput,
                o.optimized_model.throughput,
            ),
            __format_str(
                "{:.2E}".format(o.get_original_cost_per_inference()),
                "{:.2E}".format(o.get_optimized_cost_per_inference()),
            ),
        )
    print(results_summary_table)

    # Best results
    lowest_latency_template = (
        "[bold]Lowest latency[/bold]: [green]{}[/green] ({}ms, {}$/hr)"
    )
    lowest_cost_template = (
        "[bold]Lowest cost[/bold]: [green]{}[/green] ({}ms, {}$/hr)"
    )
    if experiment.result.lowest_latency is None:
        print(
            lowest_latency_template.format(
                constants.NOT_AVAILABLE_MSG,
                constants.NOT_AVAILABLE_MSG,
                constants.NOT_AVAILABLE_MSG,
            )
        )
        print(
            lowest_cost_template.format(
                constants.NOT_AVAILABLE_MSG,
                constants.NOT_AVAILABLE_MSG,
                constants.NOT_AVAILABLE_MSG,
            )
        )
    else:
        lowest_latency = experiment.result.lowest_latency
        lowest_cost = experiment.result.lowest_cost
        print(
            lowest_latency_template.format(
                lowest_latency.vm_info.sku,
                format_float(lowest_latency.optimized_model.latency_ms),
                lowest_latency.vm_info.pricing.price_hr,
            )
        )
        print(
            lowest_cost_template.format(
                lowest_cost.vm_info.sku,
                format_float(lowest_cost.optimized_model.latency_ms),
                lowest_cost.vm_info.pricing.price_hr,
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
