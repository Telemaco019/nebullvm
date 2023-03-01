import os
from pathlib import Path

DEFAULT_RAY_ADDRESS = "http://127.0.0.1:8265"

SURFER_CONFIG_BASE_DIR_PATH = Path(os.getenv("HOME"), ".config", "surfer")
SURFER_CONFIG_FILE_NAME = "config.yaml"

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

EXPERIMENTS_STORAGE_PREFIX = "nebuly/surfer/experiments"

JOB_METADATA_EXPERIMENT_NAME = "experiment_name"
