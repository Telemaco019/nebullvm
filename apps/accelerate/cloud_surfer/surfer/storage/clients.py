import abc
from pathlib import Path
from typing import Optional, List

from surfer.storage.models import StorageProvider


class StorageClient(abc.ABC):
    @staticmethod
    def from_config(config):
        if config.provider == StorageProvider.GCP.value:
            from surfer.storage import gcp
            return gcp.GCSBucketClient(config)
        if config.provider == StorageProvider.AWS.value:
            from surfer.storage import aws
            return aws.S3Client(config)
        if config.provider == StorageProvider.AZURE.value:
            from surfer.storage import azure
            return azure.BlobStorageClient(config)
        raise ValueError(f"unknown storage provider: {config.provider}")

    @abc.abstractmethod
    async def upload(
        self,
        source: Path,
        dest: Path,
        exclude_glob: Optional[str] = None,
    ):
        """
        Upload the file or directory specified by source path
        to the specified destination.

        If source is a directory, then all the files included in it
        (and in its subdirectories) are uploaded.

        Since GCS Buckets have a flat filesystem, the hierarchy of
        the directory is reflected into filenames.

        If destination is empty, the source is uploaded to the root
        of the target Bucket, otherwise "dest" is used as prefix for
        the uploaded file names.
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
    async def upload_many(
        self,
        sources: List[Path],
        dest: Path,
        exclude_glob: Optional[str] = None,
    ):
        """Upload one or more files/directories concurrently.

        If destination is empty, the source is uploaded to the
        root of the target Bucket, otherwise "dest" is used as
        prefix for the uploaded file names.

        There can be multiple upload_many(...) coroutines running
        concurrently at the same time.
        """
        pass

    @abc.abstractmethod
    async def list(self, prefix: Optional[str]) -> List[Path]:
        """List all the files in the storage

        Parameters
        ----------
        prefix: Optional[str]
            The prefix used to filter the returned files.
            If provided, only files whose name starts with the
            prefix are returned.

        Returns
        -------
        List[Path]
            The paths to the files matching the provided prefix pattern
        """
        pass

    @abc.abstractmethod
    async def get(self, path: Path) -> Optional[str]:
        """Get the content of the file at the specified path

        Parameters
        ----------
        path: Path
            The path to the file whose content is to be retrieved

        Returns
        -------
        Optional[str]
            The content of the file if it exists, None otherwise
        """
        pass

    @abc.abstractmethod
    async def delete(self, path: Path):
        """Delete the file or directory at the specified path

        Parameters
        ----------
        path: Path
            The path to the file or directory to delete

        Raises
        ------
        FileNotFoundError
            If the file or directory does not exist
        """
        pass
