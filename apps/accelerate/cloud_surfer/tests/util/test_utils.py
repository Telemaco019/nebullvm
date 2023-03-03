from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from surfer.core.schemas import ExperimentConfig


@contextmanager
def tmp_experiment_config_file() -> Path:
    with TemporaryDirectory() as d:
        tmp_file = Path(d, "config.yaml")
        with open(tmp_file, "w") as f:
            config = ExperimentConfig(
                description="test",
                data_loader_module=Path(f.name),
                model_loader_module=Path(f.name),
            )
            f.write(yaml.dump(config.dict()))
        yield tmp_file
