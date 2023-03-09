from enum import Enum

from pydantic.main import BaseModel


class StorageProvider(str, Enum):
    AZURE = "azure"
    GCP = "gcp"
    AWS = "aws"


class StorageConfig(BaseModel):
    class Config:
        frozen = True

    provider: StorageProvider
