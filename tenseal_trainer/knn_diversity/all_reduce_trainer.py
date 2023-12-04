import time
import math

import numpy as np

from utils.distance import square_euclidean_np
from utils.comm_op import all_gather
from transmission.tenseal_shapley.tenseal_all_reduce_client import AllReduceClient


class AllReduceTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.targets = targets

        unique, counts = np.unique(self.targets, return_counts=True)
        self.label_counts = dict(zip(unique, counts))
        self.server_addr = args.a_server_address
        self.client = AllReduceClient(self.server_addr, args)


    # runs fagin, and the leader checks the label
    # the leader decides whether to stop fagin when find k-nearest
    # aggregates distance
    # the leader calculates the number of this label and all samples
    # calculate I for this label
    # calculate average I for all labels

    def get_data_size(self):
        data_sum_records = []
        sent_size = received_size = 0
        for i in range(len(self.client.data_usage_records)):
            sent_size += self.client.data_usage_records[i]['sent_size']
            received_size += self.client.data_usage_records[i]['received_size']
        data = {'sent_size': sent_size, 'received_size': received_size}
        data_sum_records.append(data)
        return data_sum_records

    def transmit(self, vector):
        summed_vector = self.client.transmit(vector)
        return summed_vector

    def find_top_k(self, test_data, test_target, k):
        start_time = time.time()
        # print(">>> start find top-{} <<<".format(k))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        # print("local distance shape: {}".format(local_dist.shape))
        # print("{} local distance: {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        comm_start = time.time()
        # global_dist = sum_sqrt_all_reduce(local_dist)
        global_dist = self.transmit(local_dist)
        # print("{} global distance: {}".format(len(global_dist), global_dist[:10]))
        comm_time = time.time() - comm_start

        select_top_start = time.time()
        top_k_ids = np.argsort(global_dist)[:self.args.k]
        top_k_dist = global_dist[top_k_ids]

        local_top_k_dist = local_dist[top_k_ids]
        sum_local_top_k_dist = np.array(np.sum(local_top_k_dist, axis=0)).reshape(1,)
        # print(sum_local_top_k_dist)
        # print(local_dist.shape)
        average_dist_top_k = np.squeeze(all_gather(sum_local_top_k_dist))

        select_top_time = time.time() - select_top_start
        if self.args.rank == 0:
            print("indices of k near neighbor = {}".format(top_k_ids))
            print("distance of k near neighbor = {}".format(top_k_dist))

        # calculate label
        count_label_start = time.time()
        label_count = [0 for _ in range(self.args.n_classes)]
        for nid in top_k_ids:
            label_count[self.targets[nid]] += 1
        pred_target = np.argmax(label_count)
        pred_prob = [i / float(k) for i in label_count]
        if self.args.rank == 0:
            print("label counts = {}".format(label_count))
            print("prob of labels = {}".format(pred_prob))
        count_label_time = time.time() - count_label_start
        if self.args.rank == 0:
            print("find top-k finish: target = {}, prediction = {}, cost {:.2f} s, "
                  "compute local dist cost {:.2f} s, communication {:.2f} s, "
                  "select top-k cost {:.2f} s, count label cost {:.2f} s"
                  .format(test_target, pred_target, time.time() - start_time,
                          local_dist_time, comm_time, select_top_time, count_label_time))

        return pred_target, pred_prob, average_dist_top_k
