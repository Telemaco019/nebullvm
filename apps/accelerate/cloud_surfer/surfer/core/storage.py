from enum import Enum
from typing import Optional, List
from urllib.parse import urlparse


class Provider(str, Enum):
    AZURE = "azure"
    GCP = "gcp"
    AWS = "aws"

    @classmethod
    def list(cls) -> List[str]:
        return list(map(lambda c: c.value, cls))


class SignedURL:
    def __init__(self, url: str):
        self.provider = self._validate(url)
        self.url = url

    @staticmethod
    def _extract_provider(hostname: str) -> Optional[Provider]:
        if hostname.endswith("blob.core.windows.net"):
            return Provider.AZURE
        if hostname.endswith("storage.googleapis.com"):
            return Provider.GCP
        if hostname.endswith("amazonaws.com"):
            return Provider.AWS
        return None

    def _validate(self, url: str) -> Provider:
        if not url.startswith("https://"):
            raise ValueError("Signed URL must start with https://")
        parse_result = urlparse(url)
        provider = self._extract_provider(parse_result.hostname)
        if provider is None:
            msg = f"{url} is not a valid cloud storage signed URL. Supported storage providers are:"
            providers = [
                "Azure Storage Account: https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview",
                "Google Cloud Storage: https://cloud.google.com/storage/docs/access-control/signed-urls",
                "AWS S3: https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html",
            ]
            msg += "\n* ".join([""] + providers)
            msg += "\n"
            raise ValueError(msg)
        return provider
