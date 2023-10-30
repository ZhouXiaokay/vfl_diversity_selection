import torch
from torch import nn


class MLPBottomModel(nn.Module):
    def __init__(self, n_f_in, n_f_out):
        super().__init__()
        self.dense = nn.Linear(n_f_in, n_f_out)
        nn.init.xavier_normal_(self.dense.weight)

    def forward(self, x):
        x = self.dense(x)
        x = torch.relu(x)
        return x


class MLPTopModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.dense_1 = nn.Linear(input_size, 20, bias=False)
        nn.init.xavier_normal_(self.dense_1.weight)
        self.dense_2 = nn.Linear(20, 2, bias=False)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = torch.relu(x)

        return x


class MLP(nn.Module):
    def __init__(self, n_f):
        super().__init__()
        self.dense_1 = nn.Linear(n_f, n_f)
        nn.init.xavier_normal_(self.dense_1.weight)
        self.dense_2 = nn.Linear(n_f, 2)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, x):
        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dense_2(x)
        return x
