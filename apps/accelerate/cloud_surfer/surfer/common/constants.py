import os
from pathlib import Path

# Ray config
DEFAULT_RAY_ADDRESS = "http://127.0.0.1:8265"
JOB_METADATA_EXPERIMENT_NAME = "experiment_name"

# Surfer config
SURFER_CONFIG_BASE_DIR_PATH = Path(os.getenv("HOME"), ".config", "surfer")
SURFER_CONFIG_FILE_NAME = "config.yaml"

# Datetime
INTERNAL_DATETIME_FORMAT = "%Y-%m-%d-%H:%M:%S-%f"
DISPLAYED_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Storage
EXPERIMENTS_STORAGE_PREFIX = "nebuly/surfer/experiments"
EXPERIMENT_RESULTS_FILE_NAME = "results.json"