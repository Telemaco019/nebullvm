import asyncio
from pathlib import Path

import typer

import surfer.log
from surfer.cli.commands import cmd_experiments as cmd

app = typer.Typer()


@app.command(name="list", help="List all the experiments")
def list_experiments(
    debug: bool = typer.Option(
        False,
        help="Enable debug mode",
    ),
):
    surfer.log.setup_logger(debug)
    asyncio.run(cmd.list_experiments())


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
    surfer.log.setup_logger(debug)
    asyncio.run(cmd.submit_experiment(name, experiment_config))


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
    surfer.log.setup_logger(debug)
    typer.confirm(
        f"Are you sure you want to stop experiment {name}?",
        abort=True,
    )
    asyncio.run(cmd.stop_experiment(name))


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
    surfer.log.setup_logger(debug)
    asyncio.run(cmd.describe_experiment(name))


@app.command(
    name="delete",
    help="Delete an experiment. "
    "If the experiment is running, it will be stopped and deleted.",
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
    surfer.log.setup_logger(debug)
    typer.confirm(
        f"Are you sure you want to delete experiment {name}?",
        abort=True,
    )
    asyncio.run(cmd.delete_experiment(name))
