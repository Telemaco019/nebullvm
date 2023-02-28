import abc
from pathlib import Path
from typing import Optional, List

from surfer.storage.models import StorageProvider


class StorageClient(abc.ABC):
    @staticmethod
    def from_config(config):
        if config.provider is StorageProvider.GCP:
            from surfer.storage import gcp
            return gcp.GCSBucketClient(config)
        if config.provider is StorageProvider.AWS:
            from surfer.storage import aws
            return aws.S3Client(config)
        if config.provider is StorageProvider.AZURE:
            from surfer.storage import azure
            return azure.BlobStorageClient(config)

    @abc.abstractmethod
    async def upload(self, source: Path, dest: Path, exclude_glob: Optional[str] = None):
        pass

    @abc.abstractmethod
    async def upload_content(self, content: str, dest: Path):
        pass

    @abc.abstractmethod
    async def upload_many(self, sources: List[Path], dest: Path, exclude_glob: Optional[str] = None):
        pass
