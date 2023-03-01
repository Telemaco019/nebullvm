from pathlib import Path
from typing import List, Optional

from surfer.storage.clients import StorageClient
from surfer.storage.models import StorageConfig, StorageProvider


class AWSStorageConfig(StorageConfig):
    provider = StorageProvider.AWS


class S3Client(StorageClient):
    def __init__(self, config: AWSStorageConfig):
        raise NotImplementedError("AWS storage is not supported yet")

    async def upload(self, source: Path, dest: Path, exclude_glob: Optional[str] = None):
        pass

    async def upload_content(self, content: str, dest: Path):
        pass

    async def upload_many(self, sources: List[Path], dest: Path, exclude_glob: Optional[str] = None):
        pass