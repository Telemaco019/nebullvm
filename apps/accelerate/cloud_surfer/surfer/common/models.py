import abc
from typing import Iterable, Callable, Any, Optional, Dict, Sequence


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
    def get_precision_metric_fn(self) -> Optional[Callable[[Any, Any, Any], float]]:
        """
        Load the metric to be used for accepting or refusing
        a precision-reduction optimization proposal.

        The returned function must accept as inputs two tuples of tensors
        (produced by the baseline and the optimized model) and the related
        original labels, and it must return a float corresponding
        to the metric value.

        If None is returned, the metric
        `nebullvm.measure.compute_relative_difference` will be used as default.
        """

    @abc.abstractmethod
    def evaluate_model(
        self,
        model: Any,
        input_data: Sequence,
    ) -> Dict[str, float]:
        """
        Compute custom metrics on the given optimized model. These metrics will
        be included in the experiment results.

        The function must return a dictionary containing the metrics, where
        each key represents the metric name.

        The returned dictionary can be empty. If so, no additional metrics
        will be included in the experiment output.

        Parameters
        ----------
        model : Any
            The optimized model to evaluate.
        input_data: Sequence
            The input data to be used for evaluating the model.
        """
