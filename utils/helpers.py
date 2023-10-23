from conf import global_args_parser
global_args = global_args_parser()
SEED = global_args.seed
print(SEED)
import random
import os
import numpy as np
import torch
import collections


def seed_torch(seed=SEED):
    # print("seed: ", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_utility_key(client_attendance):
    key = 0
    for i in reversed(client_attendance):
        key = 2 * key + i
    return key


def utility_key_to_groups(key, world_size):
    client_attendance = [0] * world_size
    for i in range(world_size):
        flag = key % 2
        client_attendance[i] = flag
        key = key // 2
    return client_attendance


def stochastic_greedy(avg_dist_list, num_clients, select_clients, subsample=0.6):

    # client_list = [i for i in range(num_clients)]
    # initialize the ground set and the selected set
    V_set = set(range(num_clients))
    SUi = set()

    m = max(select_clients, int(subsample * num_clients))
    for ni in range(select_clients):
        if m < len(V_set):
            R_set = np.random.choice(list(V_set), m, replace=False)
        else:
            R_set = list(V_set)
        if ni == 0:
            marg_util = avg_dist_list[R_set].sum(1)
            # print("here",marg_util,R_set, norm_diff[:, R_set])
            i = marg_util.argmax()
            # client_max = avg_dist_list[R_set[i]]
        else:
            # client_max_R = np.maximum(client_max[:], avg_dist_list[R_set])
            # print("here", client_max_R)
            marg_util = avg_dist_list[R_set].sum(1)
            i = marg_util.argmax()
            # client_max = client_max_R[i]
        SUi.add(R_set[i])
        V_set.remove(R_set[i])
    return SUi