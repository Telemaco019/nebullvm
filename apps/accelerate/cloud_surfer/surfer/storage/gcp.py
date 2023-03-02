import asyncio
from pathlib import Path
from typing import Optional, TextIO, List

from google.cloud import storage
from google.cloud.storage import Blob

from surfer.log import logger
from surfer.storage import util
from surfer.storage.clients import StorageClient
from surfer.storage.models import StorageConfig, StorageProvider


class GCPStorageConfig(StorageConfig):
    project: str
    bucket: str
    provider = StorageProvider.GCP


class GCSBucketClient(StorageClient):

    def __init__(self, config: GCPStorageConfig):
        self.gcs_client = storage.Client(project=config.project)
        self.bucket = storage.Bucket(self.gcs_client, config.bucket)

    @staticmethod
    async def _async_upload_from_from_string(blob: Blob, content: str):
        await asyncio.get_running_loop().run_in_executor(None, blob.upload_from_string, content)

    @staticmethod
    async def _async_upload_blob_from_file(blob: Blob, content: TextIO):
        await asyncio.get_running_loop().run_in_executor(None, blob.upload_from_file, content)

    @staticmethod
    async def _async_upload_blob_from_filename(blob: Blob, filename: str):
        await asyncio.get_running_loop().run_in_executor(None, blob.upload_from_filename, filename)

    async def _upload_dir(self, source: Path, dest: Path, exclude_globs: Optional[List[str]]):
        coros = []
        for f in util.rglob(source, "*", exclude_globs):
            if f.is_file():
                full_dest_path = dest / f.relative_to(source.parent)
                blob = self.bucket.blob(full_dest_path.as_posix())
                coros.append(self._async_upload_blob_from_filename(blob, f.as_posix()))
        await asyncio.gather(*coros)

    async def _upload_file(self, source_file_path: Path, dest_path: Path):
        final_dest_path = dest_path / source_file_path.name
        logger.debug(f'\nUploading "{source_file_path}" to "{final_dest_path}"')
        blob = self.bucket.blob(final_dest_path.as_posix())
        await self._async_upload_blob_from_filename(blob, source_file_path.as_posix())

    async def upload(self, source: Path, dest: Path, exclude_globs: List[str] = None) -> Path:
        if source.is_dir():
            await self._upload_dir(source, dest, exclude_globs)
            return Path(dest, source.name)
        elif source.is_file():
            await self._upload_file(source, dest)
            return Path(dest, source.name)
        else:
            raise ValueError(f"Source path must either link a File or a Directory, got {source}")

    async def upload_many(self, sources: List[Path], dest: Path, exclude_globs: List[str] = None):
        coros = []
        for s in sources:
            if s.is_dir():
                coros.append(self._upload_dir(s, dest, exclude_globs))
            elif s.is_file():
                coros.append(self._upload_file(s, dest))
            else:
                raise ValueError(f"Source path must either link a File or a Directory, got {s}")
        await asyncio.gather(*coros)

    async def upload_content(self, content: str, dest: Path):
        blob = self.bucket.blob(dest.as_posix())
        logger.debug(f"uploading content to {dest}")
        await self._async_upload_from_from_string(blob, content)

    async def list(self, prefix: Optional[str] = None) -> List[Path]:
        def _list_blobs():
            res = []
            for blob in self.gcs_client.list_blobs(bucket_or_name=self.bucket, prefix=prefix or ""):
                res.append(blob.name)
            return res

        blob_names = await asyncio.get_running_loop().run_in_executor(None, _list_blobs)
        return [Path(n) for n in blob_names]

    async def get(self, path: Path) -> str:
        def _download_blob() -> str:
            blob = self.bucket.get_blob(path.as_posix())
            with blob.open() as f:
                return f.read()

        return await asyncio.get_running_loop().run_in_executor(None, _download_blob)
