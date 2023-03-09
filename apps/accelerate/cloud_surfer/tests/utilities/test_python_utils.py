import unittest
from pathlib import Path

import surfer.core.experiments
from surfer import ModelLoader
from surfer.storage.models import StorageConfig, StorageProvider
from surfer.utilities.python_utils import ClassLoader, load_module
from tests import _get_assets_path


def test_get_requirements__azure():
    configs = [
        StorageConfig(
            provider=StorageProvider.AZURE,
        ),
        StorageConfig(
            provider=StorageProvider.AWS,
        ),
        StorageConfig(
            provider=StorageProvider.GCP,
        ),
    ]
    for config in configs:
        requirements = surfer.core.experiments._get_job_requirements(config)
        assert len(requirements) > 0


class TestLoadModule(unittest.TestCase):
    def test_load_module__module_exists(self):
        self.assertIsNotNone(load_module(Path(__file__)))

    def test_load_module__module_does_not_exist(self):
        with self.assertRaises(ValueError):
            load_module(Path("/tmp/does_not_exist.py"))


class TestClassLoader(unittest.TestCase):
    def test_load_from_module__path_does_not_exist(self):
        loader = ClassLoader[ModelLoader](ModelLoader)
        with self.assertRaises(ValueError):
            loader.load_from_module(Path("invalid"))

    def test_load_from_module__module_without_class(self):
        loader = ClassLoader[ModelLoader](ModelLoader)
        module_path = _get_assets_path() / "empty.py"
        with self.assertRaises(ValueError):
            loader.load_from_module(module_path)

    def test_load_from_module__multiple_classes(self):
        loader = ClassLoader[ModelLoader](ModelLoader)
        module_path = _get_assets_path() / "model_loaders.py"
        loaded = loader.load_from_module(module_path)
        self.assertIsNotNone(loaded)
        self.assertTrue(issubclass(loaded, ModelLoader))
