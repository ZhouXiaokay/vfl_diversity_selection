import torch
import torch.nn as nn
import torch.utils.data


class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=1)
        # nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, x):
        x = x.float()
        x = self.linear(x)

        return x
