from typing import Iterable

import torch

from surfer import DataLoader


class BasicDataLoader(DataLoader):
    def load_data(self, *args, **kwargs) -> Iterable:
        return [
            (
                (
                    torch.randn(8, 5, 1024),
                    torch.randn(8, 8, 1024),
                    torch.randn(8, 8, 1024),
                ),
                torch.tensor([0, 1, 0, 1, 1]),
            )
            for _ in range(100)
        ]
