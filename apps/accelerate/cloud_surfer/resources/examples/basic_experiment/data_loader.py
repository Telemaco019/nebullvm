from typing import Iterable

import torch

from surfer import DataLoader


class BasicDataLoader(DataLoader):
    def load_data(self, *args, **kwargs) -> Iterable:
        return [((torch.randn(1, 3, 256, 256),), 0) for _ in range(100)]
