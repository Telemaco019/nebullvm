from typing import List

from surfer.computing.models import VMProvider

enabled_providers: List[VMProvider] = []

try:
    from surfer.computing.providers import azure

    enabled_providers.append(VMProvider.AZURE)
except ImportError as e:
    pass

try:
    from surfer.computing.providers import aws

    enabled_providers.append(VMProvider.AWS)
except ImportError:
    pass

try:
    from surfer.computing.providers import gcp

    enabled_providers.append(VMProvider.GCP)
except ImportError:
    pass
