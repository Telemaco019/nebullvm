from typing import List


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


