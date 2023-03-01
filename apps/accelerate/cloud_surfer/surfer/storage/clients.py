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
        """
        Upload the file or directory specified by source path to the specified destination.

        If source is a directory, then all the files included in it (and in its subdirectories) are uploaded.
        Since GCS Buckets have a flat filesystem, the hierarchy of the directory is reflected into
        filenames.

        If destination is empty, the source is uploaded to the root of the target Bucket,
        otherwise "dest" is used as prefix for the uploaded file names.
        """
        pass

    @abc.abstractmethod
    async def upload_content(self, content: str, dest: Path):
        """Upload the content to the file corresponding to the destination Path

        Parameters
        ----------
        content: str
            The content of the file to upload
        dest
            The path to which the file will be uploaded
        """
        pass

    @abc.abstractmethod
    async def upload_many(self, sources: List[Path], dest: Path, exclude_glob: Optional[str] = None):
        """Upload one or more files/directories concurrently.

        If destination is empty, the source is uploaded to the root of the target Bucket,
        otherwise "dest" is used as prefix for the uploaded file names.

        There can be multiple upload_many(...) coroutines running concurrently at the same time.
        """
        pass