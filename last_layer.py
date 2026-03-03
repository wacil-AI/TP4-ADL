import torch.nn as nn


class LastLayer(nn.Linear):
    def __init__(self):
        super().__init__(in_features=512, out_features=2, bias=True)


