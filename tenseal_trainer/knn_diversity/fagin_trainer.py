import time
import math

import numpy as np
import torch
import torch.distributed as dist
from multiprocessing import Process
from multiprocessing import Queue
from utils.distance import square_euclidean_np
from utils.comm_op import gather, sum_sqrt_all_reduce, sum_all_reduce, all_gather
from utils.fagin_utils import suggest_size, master_count_by_arr, master_count_fagin_group
from transmission.tenseal_shapley.tenseal_all_reduce_client import AllReduceClient


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


class FaginTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets
        self.client_list = [i for i in range(args.num_clients)]
        self.server_addr = args.a_server_address
        self.client = AllReduceClient(self.server_addr, args)

    def transmit(self, vector):

        summed_vector = self.client.transmit(vector)
        return summed_vector

    def find_top_k(self, test_data, test_target, k):
        start_time = time.time()
        if self.args.rank == 0:
            print(">>> start find top-{} <<<".format(k))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        # print("local distance size = {}, values = {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)

        # print("local dist index = {}".format(local_dist_ind[:10]))
        # print("local dist = {}".format(local_dist[local_dist_ind[:10]]))
        sort_time = time.time() - sort_start

        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size)
        if self.args.rank == 0:
            print("suggest batch size = {}".format(send_size))
        send_ind = 0

        fagin_start = time.time()
        gather_time = 0
        bc_time = 0
        count_time = 0
        top_k_ids = []
        counts = [0 for _ in range(self.n_data)]
        cur_n_top = 0
        n_iter = 0
        rank = dist.get_rank()

        while cur_n_top < self.args.k and send_ind <= self.n_data:
            gather_start = time.time()
            new_lists = gather(local_dist_ind[send_ind:min(self.n_data, send_ind + send_size)])
            gather_time += time.time() - gather_start
            send_ind += send_size
            if rank == 0:
                count_start = time.time()
                master_count_by_arr(new_lists, counts, top_k_ids, self.args.k)
                count_time += time.time() - count_start
                bc_start = time.time()
                cur_n_top = len(top_k_ids)
                dist.broadcast(torch.tensor(cur_n_top), 0)
                bc_time += time.time() - bc_start
                # print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
            else:
                bc_start = time.time()
                tmp_tensor = torch.tensor(0)
                dist.broadcast(tmp_tensor, 0)
                bc_time += time.time() - bc_start
                cur_n_top = tmp_tensor.item()
                # print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
        fagin_time = time.time() - fagin_start

        # get candidates for top-k, i.e, the instances seen so far in fagin
        candidate_start = time.time()
        n_candidate = 0
        candidate_ind = []
        if rank == 0:
            candidate_ind = [i for i, e in enumerate(counts) if e > 0]
            n_candidate = len(candidate_ind)
            # print("number of candidates = {}".format(n_candidate))
            dist.broadcast(torch.tensor(n_candidate), 0)
            dist.broadcast(torch.tensor(candidate_ind, dtype=torch.int32), 0)
        else:
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            n_candidate = tmp_tensor.item()
            # print("number of candidates = {}".format(n_candidate))
            tmp_tensor = torch.zeros([n_candidate], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            candidate_ind = tmp_tensor.tolist()
            # print("top-k candidates = {}".format(candidate_ind))
            # print("number of candidates = {}".format(n_candidate))
        candidate_time = time.time() - candidate_start

        # sync candidates for top-k, i.e, the instances seen so far in fagin
        candidate_dist_start = time.time()
        candidate_local_dist = local_dist[candidate_ind]
        # all_candidate_local_dist = all_gather(candidate_local_dist)

        # candidate_dist = sum_all_reduce(candidate_local_dist)
        dist.barrier()
        candidate_dist = self.client.transmit(candidate_local_dist)
        candidate_dist_time = time.time() - candidate_dist_start
        dist.barrier()

        # sort global distance
        select_top_start = time.time()
        sort_ids = np.argsort(candidate_dist)
        sort_dist = candidate_dist[sort_ids]
        sort_time = time.time() - sort_start

        # select top-k
        select_top_start = time.time()
        ind_k = sort_ids[:self.args.k]

        top_k_ids = np.array(candidate_ind)[ind_k]

        sorted_dist_top_k = candidate_dist[ind_k]
        select_top_time = time.time() - select_top_start
        if self.args.rank == 0:
            print("indices of k near neighbor = {}".format(top_k_ids))
            print("distance of k near neighbor = {}".format(sorted_dist_top_k))

        # calculate label
        count_label_start = time.time()
        label_count = [0 for _ in range(self.args.n_classes)]
        for j in top_k_ids:
            label_count[self.targets[j]] += 1
        pred_target = np.argmax(label_count)
        pred_prob = [i / float(k) for i in label_count]
        # print(all_candidate_local_dist)

        local_top_k_dist = local_dist[top_k_ids]
        sum_local_top_k_dist = np.array(np.sum(local_top_k_dist, axis=0)).reshape(1,)
        average_dist_top_k = np.squeeze(all_gather(sum_local_top_k_dist))

        # if self.args.rank == 0:
        #     print(all_candidate_local_dist)
        #     print(average_dist_top_k)

        if self.args.rank == 0:
            print("candidate local dist = {}".format(candidate_local_dist[:10]))
            print("candidate dist = {}".format(candidate_dist[:10]))
            print("indices of k near neighbor = {}".format(top_k_ids))
            print("distance of k near neighbor = {}".format(sorted_dist_top_k))
            # print("average distance of k near neighbor = {}".format(np.average(sorted_dist_top_k, axis=1)))
            # print(np.average(sorted_dist_top_k,axis=1))

        return pred_target, pred_prob, average_dist_top_k
