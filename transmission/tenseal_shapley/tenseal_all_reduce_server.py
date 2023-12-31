import time
from concurrent import futures
import sys
code_path = '/home/xy_li/code/vfl_diversity_selection'
sys.path.append("../../")
sys.path.append(code_path)
import numpy as np
import grpc
import tenseal as ts
from conf.args import global_args_parser


import tenseal_allreduce_data_pb2_grpc, tenseal_allreduce_data_pb2


class AllReduceServer(tenseal_allreduce_data_pb2_grpc.AllReduceServiceServicer):

    def __init__(self, address, num_clients, ctx_file):
        self.address = address
        self.num_clients = num_clients

        context_bytes = open(ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.sleep_time = 0.01

        # cache and counter for sum operation
        self.n_sum_round = 0
        self.sum_enc_vectors = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False

        print("all reduce server has been initialized")

    def reset_sum(self):
        self.sum_enc_vectors = []
        self.sum_data = []
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.sum_completed = False

    def sum_enc(self, request, context):

        server_start = time.time()

        client_rank = request.client_rank

        print(">>> server receive encrypted data from client {}, time = {} ----"
              .format(client_rank, time.asctime(time.localtime(time.time()))))

        # deserialize vector from bytes
        deser_start = time.time()
        enc_vector = ts.ckks_vector_from(self.ctx, request.msg)
        deser_time = time.time() - deser_start

        self.sum_enc_vectors.append(enc_vector)
        self.n_sum_request += 1
        # print("here 1", self.n_sum_request)
        # wait until receiving of all clients' requests
        wait_start = time.time()
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        wait_time = time.time() - wait_start

        if client_rank == 0:
            sum_start = time.time()
            summed_enc_vector = sum(self.sum_enc_vectors)
            self.sum_data.append(summed_enc_vector)
            sum_time = time.time() - sum_start
            self.sum_completed = True
        # print("here 2", client_rank)
        # print("here here", client_rank, self.sum_enc_vectors)
        sum_wait_start = time.time()
        while not self.sum_completed:
            time.sleep(self.sleep_time)
        sum_wait_time = time.time() - sum_wait_start
        # print(self.sum_data[0].decrypt()[0:10])
        # create response
        response_start = time.time()
        response = tenseal_allreduce_data_pb2.client_msg(
            client_rank=client_rank,
            msg=self.sum_data[0].serialize()
        )
        response_time = time.time() - response_start
        # print("here 3")
        # wait until creating all response
        self.n_sum_response = self.n_sum_response + 1
        # print("here 4", self.n_sum_response)
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)
        # print("here 5")

        if client_rank == 0:
            self.reset_sum()

        # wait until cache for sum is reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)

        print(">>> server finish sum_enc, cost {:.2f} s: deserialization {:.2f} s, "
              "wait for requests {:.2f} s, wait for sum {:.2f} s, create response {:.2f} s"
              .format(time.time() - server_start, deser_time,
                      wait_time, sum_wait_time, response_time))

        return response


def launch_server(address, num_clients, ctx_file):
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    servicer = AllReduceServer(address, num_clients, ctx_file)
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    tenseal_allreduce_data_pb2_grpc.add_AllReduceServiceServicer_to_server(servicer, server)

    server.add_insecure_port(address)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    args = global_args_parser()
    server_address = args.a_server_address
    # num_clients = args.num_clients
    num_clients = args.world_size
    ctx_file = args.config
    launch_server(server_address, num_clients, ctx_file)
