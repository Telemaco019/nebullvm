from typing import Iterable

from surfer import DataLoader


class BasicDataLoader(DataLoader):
    def load_data(self, *args, **kwargs) -> Iterable:
        return [
            "a photo of an astronaut riding a horse on mars",
            "a monkey eating a banana in a forest",
            "white car on a road surrounded by palm trees",
            "a fridge full of bottles of beer",
            "madara uchiha throwing asteroids against people",
        ]
