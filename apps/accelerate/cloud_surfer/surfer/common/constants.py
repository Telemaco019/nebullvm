import os
from pathlib import Path

# Ray config
DEFAULT_RAY_ADDRESS = "http://127.0.0.1:8265"
JOB_METADATA_EXPERIMENT_NAME = "experiment_name"
CLUSTER_ACCELERATOR_TYPE_PREFIX = "accelerator_type:"

# Surfer config
SURFER_CONFIG_BASE_DIR_PATH = Path(os.getenv("HOME"), ".config", "surfer")
SURFER_CONFIG_FILE_NAME = "config.yaml"

# Datetime
INTERNAL_DATETIME_FORMAT = "%Y-%m-%d-%H:%M:%S-%f"
DISPLAYED_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
HOURS_PER_YEAR = 8760

# Storage
EXPERIMENTS_STORAGE_PREFIX = "nebuly/surfer/experiments"
EXPERIMENT_RESULT_FILE_NAME = "results.json"
EXPERIMENT_MARKER_FILE_NAME = "marker.keepme"
INFERENCE_LEARNERS_DIR_NAME = "inference-learners"

# UI
NOT_AVAILABLE_MSG = "[italic]N/A[/italic]"