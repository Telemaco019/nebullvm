from pathlib import Path

import typer as typer
from rich.prompt import Confirm, Prompt

from surfer.cli.experiments import app as experiment_app
from surfer.core.config import SurferConfigManager, SurferConfig
from surfer.log import logger
from surfer.storage.models import StorageProvider, StorageConfig

app = typer.Typer(no_args_is_help=True)
app.add_typer(
    experiment_app,
    no_args_is_help=True,
    name="experiments",
    help="Manage model optimization experiments",
)


def _new_azure_storage_config() -> StorageConfig:
    try:
        from surfer.storage import azure
        sas_url = azure.URLPrompt.ask("Please enter a Storage Container SAS URL")
        return azure.AzureStorageConfig(sas_url=sas_url)
    except ImportError as e:
        raise ImportError(f'{e} - Please install "surfer[azure]" it to use Azure as storage provider')


def _new_gcp_storage_config() -> StorageConfig:
    try:
        from surfer.storage import gcp
        bucket = Prompt.ask("Insert bucket name")
        project = Prompt.ask("Insert project name")
        return gcp.GCPStorageConfig(bucket=bucket, project=project)
    except ImportError as e:
        raise ImportError(f'{e} - Please install "surfer[gcp]" it to use GCP as storage provider')


def _new_aws_storage_config():
    try:
        from surfer.storage import aws
        raise NotImplementedError("AWS storage is not yet implemented")
    except ImportError as e:
        raise ImportError(f'{e} - Please install "surfer[aws]" it to use AWS as storage provider')


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
            help="The cloud storage provider used for storing experiment data and optimized models",
        ),
):
    # Init storage config
    storage_config = None
    if storage_provider is StorageProvider.AZURE:
        storage_config = _new_azure_storage_config()
    if storage_provider is StorageProvider.AWS:
        storage_config = _new_aws_storage_config()
    if storage_provider is StorageProvider.GCP:
        storage_config = _new_gcp_storage_config()

    # Save config
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
            storage=storage_config,
        )
        config_manager.save_config(config)
        logger.info("Cloud Surfer configuration initialized", config_manager.load_config())


def main():
    app()
