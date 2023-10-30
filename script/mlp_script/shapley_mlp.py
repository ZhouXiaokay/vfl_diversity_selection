import time, math
import sys
code_path = '/home/userdata/zxk/codes/vfl_data_valuation'
sys.path.append(code_path)
from utils.helpers import seed_torch, get_utility_key, utility_key_to_groups
from conf import global_args_parser
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
seed_torch()
import argparse
import torch.distributed as dist
from trainer.mlp_shapley.shapley_mlp_trainer import ShapleyMLPTrainer

from data_loader.load_data import load_dummy_partition_with_label, load_credit_data, load_bank_data, load_mushroom_data, \
    load_covtype_data, load_adult_data, load_web_data, load_phishing_data
from torch.multiprocessing import Process
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import logging


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    # logging.basicConfig(level=logging.DEBUG,
    #                     filename=code_path + '/logs/mlp_baseline.log',
    #                     datefmt='%Y/%m/%d %H:%M:%S',
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    # logger = logging.getLogger(__name__)

    run_start = time.time()

    num_clients = args.num_clients
    # rank 0 is master
    print("rank = {}, world size = {}, pre trained = {}".format(args.rank, args.world_size, args.load_flag))
    d_name = args.dataset
    args.save_path = code_path + '/save/all_participate/mlp/' + d_name

    client_rank = args.rank
    world_size = args.world_size
    device = args.device

    dataset = load_credit_data()
    # dataset = load_bank_data()
    # dataset = load_mushroom_data()
    # dataset = load_covtype_data()
    # dataset = load_adult_data()
    # dataset = load_web_data()
    # dataset = load_phishing_data()
    data, targets = load_dummy_partition_with_label(dataset, args.num_clients, client_rank)
    # print(data[0])

    train_x, test_x, train_targets, test_targets = train_test_split(data, targets, train_size=0.8)

    train_x = torch.from_numpy(train_x).float().to(device)
    test_x = torch.from_numpy(test_x).float().to(device)
    train_targets = torch.from_numpy(train_targets).to(device)
    test_targets = torch.from_numpy(test_targets).to(device)

    n_train = train_x.shape[0]
    n_f = train_x.shape[1]
    args.n_f = n_f

    batch_size = args.batch_size
    n_batches = n_train // batch_size
    n_epochs = args.n_epochs
    utility_value = dict()
    n_utility_round = 0

    # cal utility of all groups, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, world_size)) - 1
    utility_start = time.time()
    n_utility_epochs = 0  # total used epochs
    if args.rank == 0:
        flag_msg = "dataset:{}, pretrained:{}, loss_total:{}, start_id:{}, seed:{}".format(d_name, args.load_flag,
                                                                                           args.loss_total,
                                                                                           args.start_id,
                                                                                           args.seed)
        # logger.info(flag_msg)
    for group_key in range(start_key, end_key + 1):
        seed_torch()
        group_flags = utility_key_to_groups(group_key, world_size)
        if args.rank == 0:
            print("--- compute utility of group : {} ---".format(group_flags))

        group_start = time.time()
        trainer = ShapleyMLPTrainer(args, group_flags)

        epoch_loss_lst = []
        # loss_tol = 0.1
        # epoch_tol = 3  # loss should decrease in ${epoch_tol} epochs
        epoch_tol = args.epoch_total
        loss_tol = args.loss_total
        start_id = args.start_id
        accuracy, auc = 0.0, 0.0

        for epoch_idx in range(n_epochs):

            epoch_start = time.time()
            epoch_loss = 0.

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = (batch_idx + 1) * batch_size if batch_idx < n_batches - 1 else n_train
                cur_train = train_x[start:end]
                cur_target = train_targets[start:end].long()
                batch_loss = trainer.one_iteration(cur_train, cur_target)
                epoch_loss += batch_loss
            epoch_train_time = time.time() - epoch_start
            test_start = time.time()
            pred_targets, pred_probs = trainer.predict(test_x)

            accuracy = accuracy_score(test_targets, pred_targets)
            auc = roc_auc_score(test_targets, np.array(pred_probs))
            epoch_test_time = time.time() - test_start
            if args.rank == 0:
                print(
                    ">>> epoch[{}] finish, train loss {:.6f}, cost {:.2f} s, train cost {:.2f} s, test cost {:.2f} s, "
                    "accuracy = {:.6f}, auc = {:.6f}"
                    .format(epoch_idx, epoch_loss, time.time() - epoch_start, epoch_train_time, epoch_test_time,
                            accuracy,
                            auc))
            epoch_loss_lst.append(epoch_loss)

            if epoch_idx >= start_id and len(epoch_loss_lst) > epoch_tol \
                    and min(epoch_loss_lst[:-epoch_tol]) - min(epoch_loss_lst[-epoch_tol:]) < loss_tol:
                if args.rank == 0:
                    print("!!! train loss does not decrease > {} in {} epochs, early stop !!!"
                          .format(loss_tol, epoch_tol))
                break
        n_utility_epochs += epoch_idx + 1

        utility_value[group_key] = accuracy
        n_utility_round += 1
        if args.rank == 0:
            print("compute utility of group {} cost {:.2f} s".format(group_flags, time.time() - group_start))
            group_msg = "compute utility of group {} cost {:.2f} s epoch {}".format(group_flags,
                                                                                    time.time() - group_start,
                                                                                    epoch_idx)
            # logger.info(group_msg)

    if args.rank == 0:
        print("calculate utility cost {:.2f} s, total round {}, total epochs {}"
              .format(time.time() - utility_start, n_utility_round, n_utility_epochs))

        result_msg = "calculate utility cost {:.2f} s, total round {}, total epochs {}".format(
            time.time() - utility_start,
            n_utility_round,
            n_utility_epochs)
        # logger.info(result_msg)

    if args.rank == 0:
        group_acc_sum = [0 for _ in range(args.world_size)]
        for group_key in range(start_key, end_key + 1):
            group_flags = utility_key_to_groups(group_key, world_size)
            n_participant = sum(group_flags)
            group_acc_sum[n_participant - 1] += utility_value[group_key]
            print("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))

            # logger.info("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))
        print("accuracy sum of different size: {}".format(group_acc_sum))

        # cal factorial
        factor = [1] * args.world_size
        for epoch_idx in range(1, args.world_size):
            factor[epoch_idx] = factor[epoch_idx - 1] * epoch_idx

        # shapley value of all clients
        shapley_value = [0.0] * world_size
        n_shapley_round = 0

        # cal shapley value of each
        shapley_start = time.time()
        for epoch_idx in range(world_size):
            score = 0.0
            # loop all possible groups including the current client
            start_key = 1
            end_key = int(math.pow(2, world_size)) - 1
            for group_key in range(start_key, end_key + 1):
                group_flags = utility_key_to_groups(group_key, world_size)
                group_size = sum(group_flags)
                # the current client is in the group
                if group_flags[epoch_idx] == 1 and group_size > 1:
                    u_with = utility_value[group_key]
                    group_flags[epoch_idx] = 0
                    group_key = get_utility_key(group_flags)
                    u_without = utility_value[group_key]
                    score += factor[group_size - 1] / float(factor[world_size - 1]) * (u_with - u_without)
            score /= float(math.pow(2, world_size - 1))
            shapley_value[epoch_idx] = score
            n_shapley_round += 1
        print("calculate shapley value cost {:.2f} s".format(time.time() - shapley_start))
        print("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

        shapley_ind = np.argsort(np.array(shapley_value))
        print("client ranking = {}".format(shapley_ind.tolist()[::-1]))

        # logger.info("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))
        # logger.info("client ranking = {}".format(shapley_ind.tolist()[::-1]))



def init_processes(arg, fn):
    rank = arg.rank
    size = arg.world_size
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='gloo',
                            init_method="tcp://127.0.0.1:23579",
                            rank=rank,
                            world_size=size)
    fn(arg)


if __name__ == "__main__":
    # init_processes(0, 2, run)
    processes = []
    args = global_args_parser()
    # torch.multiprocessing.set_start_method("spawn")
    # args.dataset = 'credit'
    for r in range(args.world_size):
        args.rank = r
        args.load_flag = False
        p = Process(target=init_processes, args=(args, run))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
