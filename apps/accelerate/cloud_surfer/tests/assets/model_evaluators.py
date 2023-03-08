from typing import Callable, Any
from unittest.mock import MagicMock

from surfer import ModelEvaluator


class MockedModelEvaluator(ModelEvaluator):
    def get_precision_metric_fn(self) -> Callable[[Any, Any, Any], float]:
        return MagicMock()
