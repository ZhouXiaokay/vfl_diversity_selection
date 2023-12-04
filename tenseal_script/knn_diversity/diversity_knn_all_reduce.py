import time
import sys
code_path = '/home/xy_li/code/vfl_diversity_selection'
sys.path.append(code_path)

import math
from conf import global_args_parser
import numpy as np
from torch.multiprocessing import Process

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score

# sys.path.append("../../")
from data_loader.load_data import load_dummy_partition_with_label, choose_dataset
from tenseal_trainer.knn_diversity.all_reduce_trainer import AllReduceTrainer
from utils.helpers import seed_torch, stochastic_greedy

import logging


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
    data_name = 'bank'
    dataset = choose_dataset(data_name)
    # logging.basicConfig(filename='diversity_knn_all_reduce_' + data_name+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    load_start = time.time()
    data, targets = load_dummy_partition_with_label(dataset, args.num_clients, rank)
    targets = np.int64(targets)
    # print(data[0])
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

    # cal utility of all group_keys, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1

    trainer = AllReduceTrainer(args, train_data, train_targets)

    utility_start = time.time()
    pred_targets = []
    pred_probs = []
    true_targets = []
    avg_dists = []

    for i in range(args.n_test):
        # print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        true_targets.append(cur_test_target)
        # trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_keys)
        pred_target, pred_prob, avg_dist = trainer.find_top_k(cur_test_data, cur_test_target, args.k)
        # if args.rank == 0:
        #     print(avg_dist)
        pred_targets.append(pred_target)
        pred_probs.append(pred_prob)
        avg_dists.append(avg_dist)

        one_test_time = time.time() - one_test_start

    time_cost = time.time() - utility_start
    sent_size = trainer.get_data_size()[-1]['sent_size']
    received_size = trainer.get_data_size()[-1]['received_size']

    # logging.info(f"Sent msg size is {sent_size}, received msg size is {received_size}, time cost is {time_cost}")

    print(f"sent msg size is {sent_size}, received msg size is {received_size}, time cost is {time_cost}\r\n")

    # print(trainer.get_data_size())
    avg_dists = np.average(np.array(avg_dists), axis=0)
    client_local_dist = avg_dists[:, np.newaxis]
    select_clients = stochastic_greedy(client_local_dist, args.num_clients, args.select_clients)
    # print(client_local_dist)
    if args.rank == 0:
        print("selected clients are: ", select_clients)
        print("client local dist: ", client_local_dist)
        # logging.info("Selected clients are: %s", select_clients)
        # logging.info("Client local dist: %s", client_local_dist)



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
