from typing import Iterable
from unittest.mock import MagicMock

from surfer import DataLoader


class MockedDataLoader(DataLoader):
    def load_data(self, *args, **kwargs) -> Iterable:
        return [MagicMock()]
