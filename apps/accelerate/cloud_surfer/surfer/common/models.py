import abc
from typing import Iterable, Callable, Any


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load_data(self, *args, **kwargs) -> Iterable:
        pass


class ModelLoader(abc.ABC):
    @abc.abstractmethod
    def load_model(self, *args, **kwargs) -> any:
        pass


class ModelEvaluator(abc.ABC):
    @abc.abstractmethod
    def get_precision_metric_fn(self) -> Callable[[Any, Any, Any], float]:
        pass
