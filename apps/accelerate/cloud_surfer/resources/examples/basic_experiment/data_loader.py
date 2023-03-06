from typing import Iterable

from surfer import DataLoader


class BasicDataLoader(DataLoader):
    def load_data(self, *args, **kwargs) -> Iterable:
        return []
