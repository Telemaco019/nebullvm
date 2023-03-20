from typing import Callable, Any, Sequence, Dict
from unittest.mock import MagicMock

from surfer import ModelEvaluator


class MockedModelEvaluator(ModelEvaluator):
    def get_precision_metric_fn(self) -> Callable[[Any, Any, Any], float]:
        return MagicMock()

    def evaluate_model(
        self,
        model: Any,
        input_data: Sequence,
    ) -> Dict[str, float]:
        return {}
