import unittest
from unittest.mock import MagicMock

import torch

from surfer.optimization.operations import OptimizeInferenceOp


class TestOptimizeInferenceOp(unittest.TestCase):
    def test_execute__none_model_should_raise_error(self):
        op = OptimizeInferenceOp()
        with self.assertRaises(ValueError):
            op.execute(model=None, input_data=[i for i in range(10)])

    def test_execute__empty_data_should_raise_error(self):
        op = OptimizeInferenceOp()
        with self.assertRaises(ValueError):
            op.execute(model=MagicMock(), input_data=[])

    def test_execute__unsupported_model(self):
        op = OptimizeInferenceOp()
        data = [((torch.randn(1, 3, 256, 256),), 0) for _ in range(100)]
        with self.assertRaises(TypeError):
            op.execute(model=MagicMock(), input_data=data)
