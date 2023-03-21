import unittest

from surfer.computing.models import Accelerator


class TestAccelerator(unittest.TestCase):
    def test_is_tpu(self):
        self.assertTrue(Accelerator.TPU_V2_8.is_tpu())
        self.assertTrue(Accelerator.TPU_V2_32.is_tpu())
        self.assertFalse(Accelerator.NVIDIA_TESLA_K80.is_tpu())
        self.assertFalse(Accelerator.NVIDIA_TESLA_V100.is_tpu())
        self.assertFalse(Accelerator.NVIDIA_TESLA_T4.is_tpu())
        self.assertFalse(Accelerator.NVIDIA_TESLA_P4.is_tpu())
        self.assertFalse(Accelerator.NVIDIA_TESLA_A100.is_tpu())
        self.assertFalse(Accelerator.NVIDIA_TESLA_A10G.is_tpu())

    def test_in_operator__true(self):
        items = [
            Accelerator.NVIDIA_TESLA_V100,
            Accelerator.NVIDIA_TESLA_T4,
        ]
        self.assertTrue(Accelerator.NVIDIA_TESLA_V100 in items)

    def test_in_operator__false(self):
        items = [
            Accelerator.NVIDIA_TESLA_V100,
            Accelerator.NVIDIA_TESLA_T4,
        ]
        self.assertFalse(Accelerator.NVIDIA_TESLA_K80 in items)
