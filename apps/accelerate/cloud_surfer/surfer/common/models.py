import abc
from typing import Iterable, Callable, Any


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load_data(self, *args, **kwargs) -> Iterable:
        """Load the data to be used for optimizing the model"""


class ModelLoader(abc.ABC):
    @abc.abstractmethod
    def load_model(self, *args, **kwargs) -> any:
        """Load the model to optimize"""


class ModelEvaluator(abc.ABC):
    @abc.abstractmethod
    def get_precision_metric_fn(self) -> Callable[[Any, Any, Any], float]:
        """
        Load the metric to be used for accepting or refusing
        a precision-reduction optimization proposal.
        """
