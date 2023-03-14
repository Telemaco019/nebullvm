import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from surfer.common.schemas import SurferConfig
from surfer.core.clusters import RayCluster, Accelerator, get_ray_cluster, \
    get_available_accelerators
from tests.test_utils import MockedStorageConfig


class TestRayCluster(unittest.TestCase):
    def test_get_available_accelerators_empty(self):
        cluster = RayCluster(
            config={
                "available_node_types": {},
            }
        )
        self.assertEqual(0, len(cluster.get_available_accelerators()))

    def test_get_available_accelerators_empty_config(self):
        cluster = RayCluster(
            config={}
        )
        self.assertEqual(0, len(cluster.get_available_accelerators()))

    def test_get_available_accelerators(self):
        cluster = RayCluster(
            config={
                "available_node_types": {
                    "node0": {},
                    "node1": {
                        "resources": {
                            "CPU": 1,
                            "GPU": 1,
                        },
                    },
                    "node2": {
                        "resources": {
                            f"accelerator_type:{Accelerator.NVIDIA_TESLA_V100.value}": 1,
                            "GPU": 3,
                        }
                    },
                    "node3": {
                        "resources": {
                            f"accelerator_type:{Accelerator.NVIDIA_TESLA_T4.value}": 2,
                            "GPU": 3,
                        }
                    },
                },
            }
        )
        accelerators = cluster.get_available_accelerators()
        self.assertEqual(2, len(accelerators))
        self.assertIn(Accelerator.NVIDIA_TESLA_V100, accelerators)
        self.assertIn(Accelerator.NVIDIA_TESLA_T4, accelerators)


class TestGetRayCluster(unittest.TestCase):
    def test_empty_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "file.yaml")
            with open(path, "w") as f:
                f.write("")
            surfer_config = SurferConfig(
                cluster_file=path,
                storage=MockedStorageConfig(),
            )
            cluster = get_ray_cluster(surfer_config)
            self.assertIsNotNone(cluster)


class TestGetAvailableAccelerators(unittest.TestCase):
    def test_empty_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "file.yaml")
            with open(path, "w") as f:
                f.write("foo: bar")
            surfer_config = SurferConfig(
                cluster_file=path,
                storage=MockedStorageConfig(),
            )
            accelerators = get_available_accelerators(surfer_config)
            self.assertEqual(0, len(accelerators))
