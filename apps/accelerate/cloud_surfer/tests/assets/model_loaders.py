from unittest.mock import MagicMock

from surfer import ModelLoader


class MockModelLoader(ModelLoader):
    def load_model(self, *args, **kwargs) -> any:
        return MagicMock()


class MockModelLoaderTwo(ModelLoader):
    def load_model(self, *args, **kwargs) -> any:
        return MagicMock()
