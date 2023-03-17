import torch
from torch import nn

from surfer import ModelLoader


class BasicModelLoader(ModelLoader):
    def load_model(self, *args, **kwargs) -> any:
        class ModelWrapper(torch.nn.Module):
            def __init__(self, lstm):
                super().__init__()
                self.lstm = lstm

            def forward(self, x, h, c):
                out = self.lstm(x, (h.permute(1, 0, 2), c.permute(1, 0, 2)))
                return out[0], out[1][0], out[1][1]

        return ModelWrapper(
            nn.LSTM(
                1024,  # input size
                1024,  # hidden size
                8,  # num layers
                batch_first=True,
            )
        )
