from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from surfer.common.schemas import ExperimentConfig
from surfer.storage.models import StorageConfig, StorageProvider


@contextmanager
def tmp_experiment_config_file() -> Path:
    with TemporaryDirectory() as d:
        tmp_file = Path(d, "config.yaml")
        with open(tmp_file, "w") as f:
            config = ExperimentConfig(
                description="test",
                data_loader=Path(f.name),
                model_loader=Path(f.name),
            )
            f.write(yaml.dump(config.dict()))
        yield tmp_file


class MockedStorageConfig(StorageConfig):
    provider = StorageProvider.AZURE
