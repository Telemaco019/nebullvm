import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
from urllib.parse import urlparse

import aiofiles
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob.aio import ContainerClient
from loguru import logger
from rich.prompt import Prompt, PromptType, InvalidResponse

from surfer.storage import util
from surfer.storage.clients import StorageClient
from surfer.storage.models import StorageConfig, StorageProvider

if TYPE_CHECKING:
    from azure.storage.blob.aio import BlobClient


class SignedURL:
    def __init__(self, url: str):
        self._validate(url)
        self.url = url

    @staticmethod
    def _validate(url: str):
        if not url.startswith("https://"):
            raise ValueError("Signed URL must start with https://")
        parse_result = urlparse(url)
        error_msg = (
            f"{url} is not a valid Storage Container SAS URL. "
            "Please refer to https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview "  # noqa E501
            "for more information",
        )
        if not parse_result.hostname.endswith("blob.core.windows.net"):
            raise ValueError(error_msg)


class URLPrompt(Prompt):
    @staticmethod
    def _validate(url: str):
        try:
            _ = SignedURL(url)
        except Exception as e:
            raise InvalidResponse(str(e))

    def process_response(self, value: str) -> PromptType:
        return_value = super().process_response(value)
        self._validate(return_value)
        return return_value


class AzureStorageConfig(StorageConfig):
    provider: StorageProvider = StorageProvider.AZURE
    sas_url: str


async def _upload_file(
    container: ContainerClient,
    source: Path,
    dest: Path,
):
    logger.debug("uploading {} to {}", source, dest)
    async with aiofiles.open(source, "rb") as f:
        blob = container.get_blob_client(dest.as_posix())
        await blob.upload_blob(f, overwrite=True)


class BlobStorageClient(StorageClient):
    def __init__(self, config: AzureStorageConfig):
        self.url = config.sas_url

    @asynccontextmanager
    async def __container_client(self) -> ContainerClient:
        async with ContainerClient.from_container_url(self.url) as container:
            yield container

    async def upload_content(self, content: str, dest: Path):
        logger.debug("uploading content to {}", dest)
        async with self.__container_client() as container:
            blob = container.get_blob_client(dest.as_posix())
            await blob.upload_blob(content, overwrite=True)

    async def _upload_dir(
        self,
        source: Path,
        dest: Path,
        exclude_globs: Optional[List[str]],
    ):
        coros = []
        async with self.__container_client() as container:
            for f in util.rglob(source, "*", exclude_globs):
                if f.is_file():
                    full_dest_path = dest / f.relative_to(source.parent)
                    coro = _upload_file(container, f, full_dest_path)
                    coros.append(coro)
            await asyncio.gather(*coros)

    async def upload_many(
        self,
        sources: List[Path],
        dest: Path,
        exclude_glob: Optional[str] = None,
    ):
        pass

    async def upload(
        self,
        source: Path,
        dest: Path,
        exclude_glob: Optional[str] = None,
    ) -> Path:
        if source.is_dir():
            await self._upload_dir(source, dest, exclude_glob)
            return Path(dest, source.name)
        async with self.__container_client() as container:
            await _upload_file(container, source, dest)
            return Path(dest, source.name)

    async def list(self, prefix: Optional[str]) -> List[Path]:
        res = []
        async with self.__container_client() as container:
            async for b in container.list_blobs(prefix):
                res.append(Path(b.name))
        return res

    async def get(self, path: Path) -> Optional[str]:
        try:
            async with self.__container_client() as container:
                blob_client: BlobClient = container.get_blob_client(path.as_posix())
                stream = await blob_client.download_blob(encoding="UTF-8")
                return await stream.readall()
        except ResourceNotFoundError:
            return None

    async def delete(self, path: Path):
        try:
            async with self.__container_client() as container:
                async for b in container.list_blobs(path.as_posix()):
                    logger.debug("deleting {}", b.name)
                    await container.get_blob_client(b).delete_blob()
        except ResourceNotFoundError as e:
            raise FileNotFoundError(e.message)
