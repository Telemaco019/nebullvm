from pathlib import Path

import typer as typer
from rich.prompt import Confirm, Prompt

from surfer import storage
from surfer.common.schemas import SurferConfig
from surfer.core.config import SurferConfigManager
from surfer.log import logger
from surfer.storage.models import StorageProvider, StorageConfig


def _new_azure_storage_config() -> StorageConfig:
    if StorageProvider.AZURE not in storage.enabled_providers:
        logger.error(
            "Azure storage is not enabled. "
            "Please install surfer[azure] to use Azure as storage provider"
        )
        raise typer.Exit(1)

    from surfer.storage.providers import azure

    sas_url = azure.URLPrompt.ask(
        "Please enter a Storage Container SAS URL",
    )
    return azure.AzureStorageConfig(sas_url=sas_url)


def _new_gcp_storage_config() -> StorageConfig:
    if StorageProvider.GCP not in storage.enabled_providers:
        logger.error(
            "GCP storage is not enabled. "
            "Please install surfer[gcp] to use GCP as storage provider"
        )
        raise typer.Exit(1)

    from surfer.storage.providers import gcp

    bucket = Prompt.ask("Insert bucket name")
    project = Prompt.ask("Insert project name")
    return gcp.GCPStorageConfig(bucket=bucket, project=project)


def _new_aws_storage_config():
    if StorageProvider.AWS not in storage.enabled_providers:
        logger.error(
            "AWS storage is not enabled. "
            "Please install surfer[aws] to use AWS as storage provider"
        )
        raise typer.Exit(1)
    from surfer.storage.providers import aws

    return aws.AWSStorageConfig()


def init(
    cluster_file: Path,
    storage_provider: StorageProvider,
    ray_address: str,
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
            ray_address=ray_address,
        )
        config_manager.save_config(config)
        logger.info(
            "Cloud Surfer configuration initialized",
            config_manager.load_config(),
        )
