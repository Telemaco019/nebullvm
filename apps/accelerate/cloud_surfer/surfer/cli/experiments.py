from pathlib import Path

import typer

from surfer.cli import util

app = typer.Typer()


@app.command(
    name="list",
    help="List all the experiments"
)
def list_experiments(
        cluster_config: Path = typer.Argument(
            lambda: util.must_get_cluster_config(),
            hidden=True,
        )
):
    print(cluster_config)


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
