from typing import List

from surfer.storage.models import StorageProvider

enabled_providers: List[StorageProvider] = []

try:
    from surfer.storage.providers.azure import AzureStorageConfig

    enabled_providers.append(StorageProvider.AZURE)
except ImportError as e:
    pass

try:
    from surfer.storage.providers.aws import AWSStorageConfig

    enabled_providers.append(StorageProvider.AWS)
except ImportError:
    pass

try:
    from surfer.storage.providers.gcp import GCPStorageConfig

    enabled_providers.append(StorageProvider.GCP)
except ImportError:
    pass
