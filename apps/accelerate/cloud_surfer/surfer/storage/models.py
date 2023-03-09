from enum import Enum

from pydantic.main import BaseModel


class StorageProvider(str, Enum):
    AZURE = "azure"
    GCP = "gcp"
    AWS = "aws"


class StorageConfig(BaseModel):
    class Config:
        use_enum_values = True
        frozen = True

    provider: StorageProvider
