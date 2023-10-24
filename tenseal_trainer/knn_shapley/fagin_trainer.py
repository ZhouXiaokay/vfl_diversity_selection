import time
import math

import numpy as np
import torch
import torch.distributed as dist

from utils.distance import square_euclidean_np
from utils.comm_op import gather, sum_sqrt_all_reduce
from utils.fagin_utils import suggest_size, master_count_by_arr, master_count_label, master_count_fagin_group
from transmission.tenseal_shapley.tenseal_all_reduce_client import AllReduceClient

class FaginTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets
        self.server_addr = args.a_server_address
        self.client = AllReduceClient(self.server_addr, args)

    def transmit(self, vector):
        summed_vector = self.client.transmit(vector)
        # print(summed_vector)
        return summed_vector

    def find_top_k(self, test_data, test_target, k, group_flags):
        start_time = time.time()
        if self.args.rank == 0:
            print(">>> start find top-{} <<<".format(k))
        is_attend = group_flags[self.args.rank]

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        if is_attend == 0:
            local_dist = np.zeros_like(local_dist)
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
                # master_count_label(new_lists, counts, top_k_ids, self.args.k)
                master_count_fagin_group(new_lists, counts, top_k_ids, self.args.k, group_flags)
                # master_count_label(new_lists, counts, top_k_ids, self.args.k,
                #                    self.targets, test_target)
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

        # comm candidate distance
        # print(candidate_ind)
        candidate_dist_start = time.time()
        candidate_local_dist = local_dist[candidate_ind]
        is_attend = group_flags[self.args.rank]
        if is_attend == 0:
            candidate_local_dist = np.zeros_like(candidate_local_dist)
        # candidate_dist = sum_sqrt_all_reduce(candidate_local_dist)
        candidate_dist = self.transmit(candidate_local_dist)
        candidate_dist_time = time.time() - candidate_dist_start

        # sort global distance
        select_top_start = time.time()
        sort_ids = np.argsort(candidate_dist)
        sort_dist = candidate_dist[sort_ids]
        sort_time = time.time() - sort_start

        # select top-k
        select_top_start = time.time()
        ind_k = sort_ids[:self.args.k]

        top_k_ids = np.array(candidate_ind)[ind_k]
        # top_k_ids = []
        # for i in ind_k:
        #     top_k_ids.append(candidate_ind[i])
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
        if self.args.rank == 0:
            print("label counts = {}".format(label_count))
            print("prob of labels = {}".format(pred_prob))
        count_label_time = time.time() - count_label_start

        if self.args.rank == 0:
            print("find top-k finish: target = {}, prediction = {}, total cost {:.2f} s, "
                  "comp dist cost {:.2f} s, sort cost {:.2f} s, "
                  "fagin cost {:.2f} s = gather cost {:.2f} s + broadcast cost {:.2f} s + count cost {:.2f} s, "
                  "sync top-k cost {:.2f} s, comm top-k distance cost {:.2f} s, "
                  "sort top-k cost{:.2f} s, count label cost {:.2f} s"
                  .format(test_target, pred_target, time.time() - start_time,
                          local_dist_time, sort_time, fagin_time, gather_time, bc_time, count_time,
                          candidate_dist_time, candidate_dist_time, select_top_time, count_label_time))

        return pred_target, pred_prob
