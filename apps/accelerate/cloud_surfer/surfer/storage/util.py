from pathlib import Path
from typing import Optional, List

from surfer.log import logger


def rglob(
    source: Path,
    glob: str,
    excluded_globs: Optional[List[str]],
) -> List[Path]:
    files = set(source.rglob(glob))
    if excluded_globs is not None:
        logger.debug(f"excluding globs {excluded_globs}")
        for g in excluded_globs:
            files = files - set(source.rglob(g))
    return files
