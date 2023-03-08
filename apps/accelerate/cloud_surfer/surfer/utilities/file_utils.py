import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Dict

import aiofiles

from surfer.log import logger


@asynccontextmanager
async def tmp_dir_clone(*sources: Path) -> Path:
    """Clone the provided paths in a temporary directory and yield it

    Sources path can reference either directories or files.
    In case of directories, the whole directory and its content are cloned.

    Upon exiting the context, the tmp directory and everything contained
    in it are removed.

    Example:
    >>> async with tmp_dir_clone(Path("/my-dir"), Path("file.txt")) as tmp:
    ...     print(tmp.as_posix())

    Parameters
    ----------
    sources : Path
        The paths to clone.

    Yields
    -------
    Path
        The path to the temporary directory.
    """
    async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        coros = []
        for source in sources:
            dst = tmp_dir_path
            if source.is_dir():
                dst = dst / source.name
            copy_fn = partial(
                shutil.copytree,
                src=source,
                dst=dst,
                dirs_exist_ok=True,
            )
            c = asyncio.get_event_loop().run_in_executor(None, copy_fn)
            coros.append(c)
        await asyncio.gather(*coros)
        yield tmp_dir_path


async def copy_files(*sources: Path, dst: Path):
    if not dst.is_dir():
        raise ValueError(f"destination {dst} must be a directory")
    coros = []
    for s in sources:
        copy_fn = partial(
            shutil.copy2,
            src=s,
            dst=dst,
        )
        c = asyncio.get_event_loop().run_in_executor(None, copy_fn)
        coros.append(c)
    await asyncio.gather(*coros)


class SpeedsterResultsCollector:
    def __init__(
        self,
        result_files_dir: Path = Path("."),
        result_files_regex: str = "*.json",
    ):
        """
        Parameters
        ----------
        result_files_dir: Path
            The path to the dir containing the results file
            produced by Speedster that will be collected
        result_files_regex: Path
            The regex for finding the results files produces by Speedster
        """
        self._results_file_dir = result_files_dir
        self._results_file_regex = result_files_regex

    def collect_results(self) -> Dict[str, any]:
        """Collect the results of a single Speedster run

        Returns
        -------
        Dict[str, any]
            A dictionary containing the results produced by Speedster
        """
        logger.info("collecting Nebullvm results...")
        result_riles = [
            f for f in self._results_file_dir.glob(self._results_file_regex)
        ]
        if len(result_riles) == 0:
            msg = "could not find any Nebullvm results file in path {}".format(
                self._results_file_dir
            )
            raise ValueError(msg)
        if len(result_riles) > 1:
            logger.warn(
                f"found {len(result_riles)} Nebullvm results file, "
                f"using only {result_riles[0]}"
            )
        with open(
            result_riles[0],
            "r",
        ) as res_file:
            return json.load(res_file)
