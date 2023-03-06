from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

from rich.prompt import Prompt, PromptType, InvalidResponse

from surfer.storage.clients import StorageClient
from surfer.storage.models import StorageConfig, StorageProvider


class SignedURL:
    def __init__(self, url: str):
        self._validate(url)
        self.url = url

    @staticmethod
    def _validate(url: str):
        if not url.startswith("https://"):
            raise ValueError("Signed URL must start with https://")
        parse_result = urlparse(url)
        error_msg = f"{url} is not a valid Storage Container SAS URL. " \
                    "Please refer to https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview " \
                    "for more information",
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
    provider = StorageProvider.AZURE
    sas_url: str


class BlobStorageClient(StorageClient):

    def __init__(self, config: AzureStorageConfig):
        self.config = config

    async def upload_content(self, content: str, dest: Path):
        pass

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
    ):
        pass

    async def list(self, glob: str) -> List[Path]:
        pass

    async def get(self, path: Path) -> Optional[str]:
        pass

    async def delete(self, path: Path):
        pass
