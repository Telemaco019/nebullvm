import os
from pathlib import Path

SURFER_CONFIG_BASE_DIR_PATH = Path(os.getenv("HOME"), ".config", "surfer")
SURFER_CONFIG_FILE_NAME = "config.yaml"

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
