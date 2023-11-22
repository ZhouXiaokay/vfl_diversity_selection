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
# from data_loader.data_partition import load_dummy_partition_with_label
from data_loader.load_data import load_dummy_partition_with_label, choose_dataset
from tenseal_trainer.knn_shapley.all_reduce_trainer import AllReduceTrainer
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


def encode(indices):
    max_index = max(indices)
    max_size = len(indices)
    result = [1] * max_size
    return result


def run(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    seed_torch()
    if args.rank == 0:
        print("device = {}".format(device))

    world_size = args.world_size
    rank = args.rank
    input_indices = [0, 1, 2, 3]
    data_rank = input_indices[rank]
    print("data rank is:", data_rank)

    data_name = 'phishing'
    dataset = choose_dataset(data_name)

    load_start = time.time()
    data, targets = load_dummy_partition_with_label(dataset, args.num_clients, data_rank)
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

    num_data = len(data)
    n_test = int(num_data * args.test_ratio)

    train_data = data[n_test:]
    train_targets = targets[n_test:]
    test_data = data[:n_test]
    test_targets = targets[:n_test]

    # accuracy of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    trainer = AllReduceTrainer(args, train_data, train_targets)

    # cal utility of all group_keys, group key = 1-(2^k-1)
    utility_start = time.time()
    group_flags = encode(input_indices)

    pred_targets = []
    pred_probs = []
    true_targets = []

    test_start = time.time()

    for i in range(args.n_test):
        # print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        true_targets.append(cur_test_target)

        pred_target, pred_prob = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_flags)
        pred_targets.append(pred_target)
        pred_probs.append(pred_prob)

        one_test_time = time.time() - one_test_start

        # print("one test finish: target = {}, prediction = {}, cost {} s"
        #      .format(cur_test_target, pred_target, one_test_time))
    if args.rank == 0:
        print("test {} data cost {} s".format(args.n_test, time.time() - test_start))

    accuracy = accuracy_score(true_targets, pred_targets)
    auc = roc_auc_score(true_targets, np.array(pred_probs)[:, 1])

    if args.rank == 0:
        print("calculate utility cost {:.2f} s, total round {}".format(time.time() - utility_start, n_utility_round))


def init_processes(arg, fn):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend=arg.backend,
                            init_method=arg.init_method,
                            rank=arg.rank,
                            world_size=arg.world_size)
    fn(arg)


if __name__ == '__main__':

    processes = []
    # torch.multiprocessing.set_start_method("spawn")
    args = global_args_parser()
    # args.dataset = 'libsvm-a8a'
    # args.loss_total = 0.01
    # args.seed = 2023
    for r in range(args.world_size):
        args.rank = r
        p = Process(target=init_processes, args=(args, run))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
