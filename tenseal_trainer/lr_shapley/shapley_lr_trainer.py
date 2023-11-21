import torch
from torch.nn import BCELoss
from model.LR import LR
import torch.optim as optim
from utils.comm_op import sum_all_reduce_tensor
from utils.helpers import seed_torch
from transmission.tenseal_shapley.tenseal_all_reduce_client import AllReduceClient


class ShapleyLRTrainer(object):
    def __init__(self, args, group_flags):
        self.args = args
        self.group_flags = group_flags
        self.is_attend = self.group_flags[self.args.rank]
        self.n_f = self.args.n_f
        # seed_torch()
        self.lr = LR(self.n_f).to(args.device)
        self.criterion = BCELoss()
        self.optimizer = optim.Adam(self.lr.parameters(), lr=1e-3)
        self.rank = args.rank
        self.server_addr = args.a_server_address
        self.client = AllReduceClient(self.server_addr, args)

    def transmit(self, vector):
        vector = vector.squeeze()
        np_vector = vector.detach().numpy()
        summed_vector = self.client.transmit(np_vector)
        summed_tensor = torch.from_numpy(summed_vector).unsqueeze(dim=1)

        return summed_tensor

    def one_iteration(self, x, y):

        partial_z = self.lr(x) if self.is_attend == 1 else torch.zeros(x.shape[0])
        sum_z = sum_all_reduce_tensor(partial_z)
        sum_z = self.transmit(partial_z)
        # sum_z = sum_all_reduce_tensor(partial_z)
        sum_z.requires_grad = True
        h = torch.sigmoid(sum_z)
        y = y.double()
        loss = torch.zeros(1)
        if self.is_attend == 1:
            loss = self.criterion(h, y)
            loss.backward()

            self.optimizer.zero_grad()
            partial_z.backward(sum_z.grad)
            self.optimizer.step()

        global_loss = loss.float().detach()
        global_loss = sum_all_reduce_tensor(global_loss)
        return global_loss.item() / self.args.world_size

    def predict(self, x):
        partial_z = self.lr(x) if self.is_attend == 1 else torch.zeros(x.shape[0])
        sum_z = sum_all_reduce_tensor(partial_z)
        pos_prob = torch.sigmoid(sum_z).squeeze().detach().numpy()
        pred = (pos_prob > 0.5).astype(int)
        return pred, pos_prob
