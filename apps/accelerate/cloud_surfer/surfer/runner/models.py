import abc
from typing import Iterable, Dict


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
    def evaluate_model(self, model, *args, **kwargs) -> Dict[str, any]:
        pass


class DefaultModelEvaluator(ModelEvaluator):
    def evaluate_model(self, model, *args, **kwargs) -> Dict[str, any]:
        pass
