import functools
from typing import List


@functools.cache
def get_requirements() -> List[str]:
    import typer

    requirements = [
        f"typer=={typer.__version__}",
    ]

    # Add GCP storage dependency
    try:
        from surfer.storage import gcp

        requirements.append(f"google-cloud-storage=={gcp.storage.__version__}")
    except ImportError:
        pass

    # Add Azure storage dependency
    # TODO

    # Add AWS storage dependency
    # TODO

    return requirements


