import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp

from surfer import ModelLoader
from surfer.core.util import (
    RandomGenerator,
    load_module,
    ClassLoader,
    tmp_dir_clone,
)


def _get_assets_path() -> Path:
    return Path(__file__).parent.parent / Path("assets")


class TestRandomGenerator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = RandomGenerator()

    def test_random_mnemonic__generated_words_all_different(self):
        generated = []
        for _ in range(5):
            generated.append(self.generator.random_mnemonic())
        self.assertEqual(len(generated), len(set(generated)))

    def test_random_mnemonic__n_words(self):
        n_words = 2
        mnemonic = self.generator.random_mnemonic(n_words=n_words)
        self.assertEqual(
            n_words, len(mnemonic.split(self.generator.separator))
        )


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


class TestTmpDirClone(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.trees_to_remove = []

    def tearDown(self) -> None:
        super().tearDown()
        for tree in self.trees_to_remove:
            shutil.rmtree(tree)

    async def test_source_does_not_exist(self):
        with self.assertRaises(FileNotFoundError):
            async with tmp_dir_clone(Path("/tmp/invalid")):
                pass

    async def test_no_sources(self):
        async with tmp_dir_clone() as f:
            self.assertTrue(f.exists())
            self.assertTrue(f.is_dir())

    async def test_multiple_sources(self):
        tmp_dir_1 = Path(mkdtemp())
        tmp_dir_1_file_1 = tmp_dir_1 / Path("test.txt")
        tmp_dir_1_file_1.touch()
        tmp_dir_2 = Path(mkdtemp())
        tmp_dir_2_dir_1 = Path(mkdtemp(dir=tmp_dir_2))

        async with tmp_dir_clone(tmp_dir_1, tmp_dir_2) as tmp:
            files = set(tmp.rglob("*"))
            self.assertEqual(4, len(files))
            self.assertTrue((tmp / tmp_dir_1).exists())
            self.assertTrue((tmp / tmp_dir_2).exists())
            self.assertTrue((tmp / tmp_dir_1_file_1).exists())
            self.assertTrue((tmp / tmp_dir_2_dir_1).exists())

        # Cleanup
        self.trees_to_remove.append(tmp_dir_1)
        self.trees_to_remove.append(tmp_dir_2)
