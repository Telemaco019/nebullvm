import inspect
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from types import ModuleType
from typing import Generic, Type, List, TypeVar

from surfer.log import logger

_T = TypeVar("_T")


class ClassLoader(Generic[_T]):
    def __init__(self, cls):
        """
        Parameters
        ----------
        cls : _T
            The searched class. The ClassLoader will load classes
            of this type, or any class that extends it.
        """
        self._cls = cls

    def _matches_searched_class(self, cls: Type):
        return inspect.isclass(cls) and self._cls in cls.__bases__

    def _load_classes(self, module_path: Path) -> List[Type[_T]]:
        loaded = load_module(module_path)
        searched_class_tuples = inspect.getmembers(
            loaded, predicate=self._matches_searched_class
        )
        return [t[1] for t in searched_class_tuples]

    def load_from_module(self, module_path: Path) -> Type[_T]:
        """Load the searched class from the provided module

        Return the searched class loaded from the local
        Python module located at the path provided as arg

        Fields
        -------
        module_path : Path
            The path to the Python module containing the class to load

        Raises
        -------
        ValueError
            If the provided path specified does not exist,
            or it does not correspond to a Python module
            containing any searched class.
        """
        loaded_classes = self._load_classes(module_path)
        if len(loaded_classes) == 0:
            msg = "module {} does not contain any {} class.".format(
                module_path,
                self._cls.__name__,
            )
            raise ValueError(msg)
        if len(loaded_classes) > 1:
            logger.warn(
                "multiple {} classes found in {}, using {}".format(
                    self._cls.__name__,
                    module_path,
                    loaded_classes[0],
                )
            )
        return loaded_classes[0]


def load_module(module_path: Path) -> ModuleType:
    if module_path.exists() is False:
        raise ValueError(f"could not find module {module_path}")
    spec_mod = spec_from_file_location("imported_module", module_path)
    loaded_module = module_from_spec(spec_mod)
    spec_mod.loader.exec_module(loaded_module)
    return loaded_module
