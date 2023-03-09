import torchvision.models as models

from surfer import ModelLoader


class BasicModelLoader(ModelLoader):
    def load_model(self, *args, **kwargs) -> any:
        return models.resnet50()
