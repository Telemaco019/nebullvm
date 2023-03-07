from pathlib import Path


def _get_assets_path() -> Path:
    return Path(__file__).parent / Path("assets")
