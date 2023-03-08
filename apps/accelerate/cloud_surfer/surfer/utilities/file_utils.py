import asyncio
import shutil
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import aiofiles


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
