import pathlib
from datetime import datetime
from importlib.util import spec_from_file_location, module_from_spec
from types import ModuleType

from random_word import RandomWords

from surfer.core import constants


class RandomGenerator:
    def __init__(self, random_words_generator: RandomWords = RandomWords(), separator="-"):
        self.__random_words_generator = random_words_generator
        self.separator = separator

    def random_mnemonic(self, n_words=3):
        return self.separator.join([self.__random_words_generator.get_random_word() for _ in range(n_words)])


def load_module(module_path: pathlib.Path) -> ModuleType:
    if module_path.exists() is False:
        raise ValueError(f"could not find module {module_path}")
    spec_mod = spec_from_file_location("imported_module", module_path)
    loaded_module = module_from_spec(spec_mod)
    spec_mod.loader.exec_module(loaded_module)
    return loaded_module


def format_datetime(dt: datetime) -> str:
    return dt.strftime(constants.DATETIME_FORMAT)
