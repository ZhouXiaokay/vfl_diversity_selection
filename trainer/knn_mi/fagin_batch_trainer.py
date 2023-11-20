import time
import math

import numpy as np
import torch
import torch.distributed as dist

from utils.distance import square_euclidean_np
from utils.comm_op import gather, sum_sqrt_all_reduce, sum_all_reduce
from utils.fagin_utils import suggest_size, master_count_by_arr, master_count_fagin_group


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


class FaginBatchTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets
        unique, counts = np.unique(self.targets, return_counts=True)
        self.label_counts = dict(zip(unique, counts))

    @staticmethod
    def digamma(x):
        return math.log(x, math.e) - 0.5 / x

    def find_top_k(self, test_data, test_target, k, group_keys):
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

        # for each group cal its global distance
        group_candidate_dist_list = []
        for key in group_keys:
            group_flags = utility_key_to_groups(key, self.args.world_size)
            group_local_dist = group_flags[rank] * candidate_local_dist
            group_dist = sum_all_reduce(group_local_dist)
            group_candidate_dist_list.append(group_dist)
        group_candidate_dist = np.array(group_candidate_dist_list)

        # sort group distance
        select_top_start = time.time()
        groups_sort_ids = np.argsort(group_candidate_dist, axis=1)
        sort_time = time.time() - sort_start

        # select top-k for groups
        # select_top_start = time.time()
        groups_ind_k = groups_sort_ids[:, :self.args.k]
        groups_top_k_ids = np.tile(np.array(candidate_ind), (len(group_keys), 1))[:, groups_ind_k][0, :, :]
        sorted_dist_top_k = group_candidate_dist[:, groups_ind_k][0, :, :]
        # average_dist_top_k = np.average(sorted_dist_top_k, axis=1)

        # calculate label
        count_label_start = time.time()

        pred_target = []
        pred_prob = []
        client_mi_values = np.zeros(self.args.world_size)

        for ids in range(len(group_keys)):
            group_flags = utility_key_to_groups(ids, self.args.world_size)

            cur_label_top_k_ids = []
            cur_label_count = 0
            all_label_count = 0
            for i in range(n_candidate):

                candidate_id = groups_sort_ids[ids][i]
                all_label_count += 1
                if self.targets[candidate_id] == test_target:
                    cur_label_top_k_ids.append(candidate_id)
                    cur_label_count += 1
                    if cur_label_count == k:
                        break
            N = len(self.data)
            N_i = self.label_counts[test_target]
            m_i = all_label_count

            mi_value = self.digamma(N) - self.digamma(N_i) + self.digamma(k) - self.digamma(m_i)

            client_mi_values += np.array(group_flags) * mi_value

        return client_mi_values

        # label_count = [0 for _ in range(self.args.n_classes)]
        # for j in ids:
        #     label_count[self.targets[j]] += 1
        # pred_target.append(np.argmax(label_count))
        # pred_prob.append([i / float(k) for i in label_count])

        # if self.args.rank == 0:
        #     print("label counts = {}".format(label_count))
        #     print("prob of labels = {}".format(pred_prob))
        count_label_time = time.time() - count_label_start
        #
        # return pred_target, pred_prob, average_dist_top_k
