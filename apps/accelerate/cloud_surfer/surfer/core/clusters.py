import enum
from typing import List

import yaml

from surfer.common import constants
from surfer.common.schemas import SurferConfig


class Accelerator(str, enum.Enum):
    NVIDIA_TESLA_V100 = "V100", "NVIDIA Tesla V100"
    NVIDIA_TESLA_P100 = "P100", "NVIDIA Tesla P100"
    NVIDIA_TESLA_T4 = "T4", "NVIDIA Tesla T4"
    NVIDIA_TESLA_P4 = "P4", "NVIDIA Tesla P4"
    NVIDIA_TESLA_K80 = "K80", "NVIDIA Tesla K80"
    NVIDIA_TESLA_A100 = "A100", "NVIDIA Tesla A100"
    NVIDIA_TESLA_A10G = "A10G", "NVIDIA Tesla A10G"

    def __new__(cls, *values):
        obj = super().__new__(cls)
        obj._value_ = values[0]
        obj.display_name = values[1]
        return obj


class RayCluster:
    def __init__(self, config: dict):
        self.config = config

    def get_available_accelerators(self) -> List[Accelerator]:
        accelerators = []
        for _, node in self.config["available_node_types"].items():
            resources = node.get("resources", None)
            if resources is None:
                continue
            for r in resources:
                parts = r.split(constants.CLUSTER_ACCELERATOR_TYPE_PREFIX)
                if len(parts) > 1:
                    accelerators.append(Accelerator(parts[1]))
        return accelerators


def get_available_accelerators(config: SurferConfig) -> List[Accelerator]:
    with open(config.cluster_file, "r") as f:
        cluster_config = yaml.safe_load(f.read())
        cluster = RayCluster(cluster_config)
        return cluster.get_available_accelerators()
