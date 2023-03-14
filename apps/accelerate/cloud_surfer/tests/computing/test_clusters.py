import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from surfer.common.schemas import SurferConfig
from surfer.computing.clusters import (
    RayCluster,
    Accelerator,
    get_ray_cluster,
    get_available_accelerators,
)
from tests.test_utils import MockedStorageConfig


class TestRayCluster(unittest.TestCase):
    def test_init__missing_provider(self):
        with self.assertRaises(ValueError):
            RayCluster(config={})

    def test_init__missing_provider_type(self):
        with self.assertRaises(ValueError):
            RayCluster(config={"provider": {}})

    def test_init__unknown_provider(self):
        with self.assertRaises(ValueError):
            RayCluster(config={"provider": {"type": "unknown"}})

    def test_get_nodes_empty_node_types(self):
        cluster = RayCluster(
            config={
                "provider": {
                    "type": "azure",
                },
                "available_node_types": {},
            }
        )
        self.assertEqual(0, len(cluster.get_nodes()))

    def test_get_nodes__invalid_node_config(self):
        cluster = RayCluster(
            config={
                "provider": {"type": "azure"},
                "available_node_types": {
                    "node0": {
                        "resources": {
                            "CPU": 1,
                            "GPU": 1,
                            f"accelerator_type:{Accelerator.NVIDIA_TESLA_K80.value}": 1,
                        },
                        "node_config": {},
                    },
                },
            }
        )
        with self.assertRaises(ValueError):
            cluster.get_nodes()

    def test_get_nodes(self):
        cluster = RayCluster(
            config={
                "provider": {"type": "azure"},
                "available_node_types": {
                    "node0": {},
                    "node1": {
                        "resources": {
                            "CPU": 1,
                            "GPU": 1,
                            f"accelerator_type:{Accelerator.NVIDIA_TESLA_K80.value}": 1,
                        },
                        "node_config": {
                            "InstanceType": "m5.large",
                        },
                    },
                    "node2": {
                        "resources": {
                            f"accelerator_type:{Accelerator.NVIDIA_TESLA_V100.value}": 1,
                            "GPU": 3,
                        },
                        "node_config": {
                            "azure_arm_parameters": {
                                "vmSize": "Standard_1",
                            }
                        },
                    },
                    "node3": {
                        "resources": {
                            f"accelerator_type:{Accelerator.NVIDIA_TESLA_T4.value}": 2,
                            "GPU": 3,
                        }
                    },
                    "node4": {
                        "resources": {
                            "CPU": 1,
                            "GPU": 1,
                            f"accelerator_type:{Accelerator.NVIDIA_TESLA_K80.value}": 1,
                        },
                        "node_config": {
                            "machineType": "n1-standard-1",
                        },
                    },
                },
            }
        )
        nodes = cluster.get_nodes()
        accelerators = [node.accelerator for node in nodes]
        vm_sizes = [node.vm_size for node in nodes]
        self.assertEqual(3, len(nodes))
        # Check accelerators
        self.assertIn(Accelerator.NVIDIA_TESLA_V100, accelerators)
        self.assertIn(Accelerator.NVIDIA_TESLA_K80, accelerators)
        # Check VM sizes
        self.assertIn("m5.large", vm_sizes)
        self.assertIn("Standard_1", vm_sizes)
        self.assertIn("n1-standard-1", vm_sizes)


class TestGetRayCluster(unittest.TestCase):
    def test_empty_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "file.yaml")
            with open(path, "w") as f:
                f.write("{}")
            surfer_config = SurferConfig(
                cluster_file=path,
                storage=MockedStorageConfig(),
            )
            with self.assertRaises(ValueError):
                get_ray_cluster(surfer_config)


class TestGetAvailableAccelerators(unittest.TestCase):
    def test_empty_config(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "file.yaml")
            with open(path, "w") as f:
                f.write("{}")
            surfer_config = SurferConfig(
                cluster_file=path,
                storage=MockedStorageConfig(),
            )
            with self.assertRaises(ValueError):
                get_available_accelerators(surfer_config)
