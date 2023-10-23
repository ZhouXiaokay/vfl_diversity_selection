import time
import sys
import math
from conf import global_args_parser
import numpy as np
from torch.multiprocessing import Process

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score

# sys.path.append("../../")
from data_loader.load_data import load_dummy_partition_with_label, load_credit_data
from trainer.knn_shapley.fagin_batch_trainer import FaginBatchTrainer
from utils.helpers import seed_torch


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


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


def run(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    seed_torch()
    print("device = {}".format(device))

    world_size = args.world_size
    rank = args.rank

    # file_name = "{}/{}_{}".format(args.root, rank, world_size)
    # print("read file {}".format(file_name))
    dataset = load_credit_data()
    load_start = time.time()
    data, targets = load_dummy_partition_with_label(dataset, args.num_clients, rank)
    if args.rank == 0:
        print("load data part cost {} s".format(time.time() - load_start))
    n_data = len(data)
    if args.rank == 0:
        print("number of data = {}".format(n_data))

    # shuffle the data to split train data and test data
    shuffle_ind = np.arange(n_data)
    np.random.shuffle(shuffle_ind)
    if args.rank == 0:
        print("test data indices: {}".format(shuffle_ind[:args.n_test]))
    data = data[shuffle_ind]
    targets = targets[shuffle_ind]

    train_data = data[args.n_test:]
    train_targets = targets[args.n_test:]
    test_data = data[:args.n_test]
    test_targets = targets[:args.n_test]

    # accuracy of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    # cal utility of all group_keys, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1
    group_keys = [i for i in range(start_key, end_key + 1)]
    trainer = FaginBatchTrainer(args, train_data, train_targets)

    utility_start = time.time()
    pred_targets = []
    pred_probs = []
    true_targets = []

    for i in range(args.n_test):
        # print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        true_targets.append(cur_test_target)
        # trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_keys)
        pred_target, pred_prob = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_keys)
        # if args.rank == 0:
        #     print(pred_target)
        pred_targets.append(pred_target)
        pred_probs.append(pred_prob)

        one_test_time = time.time() - one_test_start
    pred_targets = np.array(pred_targets)
    pred_probs = np.array(pred_probs)
    true_targets = np.array(true_targets)
    # print(group_keys)
    for key in group_keys:
        accuracy = accuracy_score(true_targets, pred_targets[:, key - 1])
        utility_value[key] = accuracy

    group_acc_sum = [0 for _ in range(args.world_size)]
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        n_participant = sum(group_flags)
        group_acc_sum[n_participant - 1] += utility_value[group_key]
        if args.rank == 0:
            print("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))
    if args.rank == 0:
        print("accuracy sum of different size: {}".format(group_acc_sum))

    # cal factorial
    factor = [1] * args.world_size
    for i in range(1, args.world_size):
        factor[i] = factor[i - 1] * i

    # shapley value of all clients
    shapley_value = [0.0] * world_size
    n_shapley_round = 0

    # cal shapley value of each
    shapley_start = time.time()
    for i in range(world_size):
        score = 0.0
        # loop all possible group_keys including the current client
        start_key = 1
        end_key = int(math.pow(2, world_size)) - 1
        for group_key in range(start_key, end_key + 1):
            group_flags = utility_key_to_groups(group_key, world_size)
            group_size = sum(group_flags)
            # the current client is in the group
            if group_flags[i] == 1 and group_size > 1:
                u_with = utility_value[group_key]
                group_flags[i] = 0
                group_key = get_utility_key(group_flags)
                u_without = utility_value[group_key]
                score += factor[group_size - 1] / float(factor[world_size - 1]) * (u_with - u_without)
        score /= float(math.pow(2, world_size - 1))
        shapley_value[i] = score
        n_shapley_round += 1
    if args.rank == 0:
        print("calculate shapley value cost {:.2f} s".format(time.time() - shapley_start))
        print("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

    shapley_ind = np.argsort(np.array(shapley_value))
    if args.rank == 0:
        print("client ranking = {}".format(shapley_ind.tolist()[::-1]))


def init_processes(arg, fn):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend=arg.backend,
                            init_method=arg.init_method,
                            rank=arg.rank,
                            world_size=arg.world_size)
    fn(arg)


if __name__ == '__main__':

    processes = []
    args = global_args_parser()
    for r in range(args.world_size):
        args.rank = r
        p = Process(target=init_processes, args=(args, run))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
