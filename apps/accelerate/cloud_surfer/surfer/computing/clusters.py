from dataclasses import dataclass
from typing import List, Dict, Optional

import yaml

from surfer.common import constants
from surfer.common.schemas import SurferConfig
from surfer.computing.models import Accelerator, VMProvider


@dataclass
class ClusterNode:
    vm_size: str
    accelerator: Accelerator


class RayCluster:
    def __init__(self, config: dict):
        self.config = config
        self.provider = self.__get_provider(config)

    @staticmethod
    def __get_provider(config) -> VMProvider:
        provider = config.get("provider", None)
        if provider is None:
            raise ValueError("missing field 'provider' in cluster config")
        provider_type = provider.get("type", None)
        if provider_type is None:
            raise ValueError("missing field 'provider.type' in cluster config")
        return VMProvider(provider_type)

    @staticmethod
    def __get_instance_type(node_config: Dict[str, any]):
        # aws
        if "InstanceType" in node_config:
            return node_config["InstanceType"]
        # gcp
        if "machineType" in node_config:
            return node_config["machineType"]
        # azure
        if (
            "azure_arm_parameters" in node_config
            and "vmSize" in node_config["azure_arm_parameters"]
        ):
            return node_config["azure_arm_parameters"]["vmSize"]
        raise ValueError(f"node config is not valid: {node_config}")

    @staticmethod
    def __extract_accelerator(resource: str) -> Optional[Accelerator]:
        parts = resource.split(constants.CLUSTER_ACCELERATOR_TYPE_PREFIX)
        if len(parts) > 1:
            return Accelerator(parts[1])
        try:
            return Accelerator(resource)
        except ValueError:
            return None

    def get_nodes(self) -> List[ClusterNode]:
        nodes: List[ClusterNode] = []
        node_types = self.config.get("available_node_types", {})
        for _, node_type in node_types.items():
            resources = node_type.get("resources", None)
            node_config = node_type.get("node_config", None)
            if resources is None:
                continue
            if node_config is None:
                continue
            for r in resources:
                accelerator = self.__extract_accelerator(r)
                if accelerator is None:
                    continue
                if accelerator.is_tpu():
                    instance_type = "{} Runtime {}".format(
                        accelerator.display_name,
                        node_config["runtimeVersion"],
                    )
                else:
                    instance_type = self.__get_instance_type(node_config)
                nodes.append(ClusterNode(instance_type, accelerator))
        return nodes


def get_ray_cluster(surfer_config: SurferConfig) -> RayCluster:
    with open(surfer_config.cluster_file, "r") as f:
        cluster_config = yaml.safe_load(f.read())
        return RayCluster(cluster_config)


def get_available_accelerators(config: SurferConfig) -> List[Accelerator]:
    return [n.accelerator for n in get_ray_cluster(config).get_nodes()]
