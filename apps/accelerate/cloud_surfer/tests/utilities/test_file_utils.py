import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp

from surfer.utilities.file_utils import tmp_dir_clone


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
