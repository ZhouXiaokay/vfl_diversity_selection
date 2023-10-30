import torch
from torch.nn.functional import cross_entropy
from model.MLP import MLPBottomModel, MLPTopModel
import torch.optim as optim
from utils.helpers import seed_torch
from utils.comm_op import sum_all_reduce_tensor, sum_all_gather_tensor

# from utils import sum_all_gather, sum_all_reduce, seed_torch, get_selected_model_params_concat_layer, \
#     split_concat_layer_params


class ShapleyMLPTrainer(object):
    def __init__(self, args, group_flags):
        self.args = args
        self.rank = args.rank
        self.group_flags = group_flags
        self.group_size = sum(self.group_flags)
        self.is_attend = self.group_flags[self.args.rank]
        self.n_f = self.args.n_f
        self.n_bottom_out = self.args.n_bottom_out
        self.bottom_model = MLPBottomModel(n_f_in=self.n_f, n_f_out=self.n_bottom_out).to(args.device)
        seed_torch()
        self.top_model = MLPTopModel(input_size=self.n_bottom_out * self.group_size).to(args.device)
        self.criterion = cross_entropy
        self.bottom_optimizer = optim.Adam(self.bottom_model.parameters(), lr=1e-3)
        self.top_optimizer = optim.Adam(self.top_model.parameters(), lr=1e-3)
        self.group_rank = self.__init_group_rank() if self.is_attend == 1 else -1

    def __init_group_rank(self):
        group_rank = 0
        for i in range(self.rank):
            if self.group_flags[i] == 1:
                group_rank += 1
        return group_rank

    def __split_bottom_grads(self, bottom_grads):
        shape_list = [self.n_bottom_out] * self.group_size
        return list(torch.split(bottom_grads, shape_list, dim=-1))

    def __select_bottom_out(self, concat_all):
        shape_list = [self.n_bottom_out] * self.args.world_size
        bottom_list = torch.split(concat_all, shape_list, dim=1)
        attend_list = []
        for rank in range(self.args.world_size):
            if self.group_flags[rank] == 1:
                attend_list.append(bottom_list[rank])
        concat_bottom = torch.concat(attend_list, dim=1)
        return concat_bottom

    def one_iteration(self, x, y):

        partial_z = self.bottom_model(x) if self.is_attend == 1 else torch.zeros((x.shape[0], self.n_bottom_out))
        all_bottom = sum_all_gather_tensor(partial_z)
        loss = torch.zeros(1)

        if self.is_attend == 1:
            concat_bottom = self.__select_bottom_out(all_bottom)
            # print(all_bottom)
            concat_bottom.requires_grad = True
            pred = self.top_model(concat_bottom)
            loss = self.criterion(pred, y)

            self.top_optimizer.zero_grad()
            loss.backward()
            self.top_optimizer.step()

            all_grads = concat_bottom.grad
            bottom_grad = self.__split_bottom_grads(all_grads)[self.group_rank]

            self.bottom_optimizer.zero_grad()
            partial_z.backward(bottom_grad)
            self.bottom_optimizer.step()

        # print(self.bottom_model.dense.weight[0])

        global_loss = loss.detach()
        global_loss = sum_all_reduce_tensor(global_loss)
        return global_loss.item() / self.args.world_size
        # return global_loss.item() / self.group_size

    def predict(self, x):
        partial_z = self.bottom_model(x) if self.is_attend == 1 else torch.zeros((x.shape[0], self.n_bottom_out))
        all_bottom = sum_all_gather_tensor(partial_z)
        pred = torch.zeros((x.shape[0], 2))
        if self.is_attend == 1:
            concat_bottom = self.__select_bottom_out(all_bottom)
            pred = self.top_model(concat_bottom)
        pred = sum_all_reduce_tensor(pred) / self.group_size
        pos_prob = torch.softmax(pred, dim=1).max(dim=1).values.detach().numpy()
        pred = torch.softmax(pred, dim=1).max(dim=1).indices.detach().numpy()

        return pred, pos_prob

    def save(self, save_path):
        torch.save(self.bottom_model.state_dict(),
                   save_path + '/bottom_model_{0}_seed_{1}.pth'.format(self.rank, self.args.seed))
        torch.save(self.top_model.state_dict(),
                   save_path + '/top_model_{0}_seed_{1}.pth'.format(self.rank, self.args.seed))

