from surfer import ModelLoader


class MockModelLoader(ModelLoader):
    def load_model(self, *args, **kwargs) -> any:
        pass


class MockModelLoaderTwo(ModelLoader):
    def load_model(self, *args, **kwargs) -> any:
        pass
