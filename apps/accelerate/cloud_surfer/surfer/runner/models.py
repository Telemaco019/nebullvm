import abc
from typing import Iterable, Dict, List


class RayCluster:
    def __init__(self, config):
        self.config = config

    def get_available_accelerators(self) -> List[str]:
        accelerators = []
        for node in self.config["available_node_types"]:
            resources = node.get("resources", None)
            if resources is None:
                continue
            for r in resources:
                parts = r.split(ACCELERATOR_TYPE_PREFIX)
                if len(parts) > 1:
                    accelerators.append(parts[1])
        return accelerators


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load_data(self, *args, **kwargs) -> Iterable:
        pass


class ModelLoader(abc.ABC):
    @abc.abstractmethod
    def load_model(self, *args, **kwargs) -> any:
        pass


class ModelEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate_model(self, model, *args, **kwargs) -> Dict[str, any]:
        pass


class DefaultModelEvaluator(ModelEvaluator):
    def evaluate_model(self, model, *args, **kwargs) -> Dict[str, any]:
        pass
