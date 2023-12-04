import time

import grpc
import numpy as np
import tenseal as ts

import sys
from transmission.tenseal_shapley import tenseal_allreduce_data_pb2_grpc, tenseal_allreduce_data_pb2


class AllReduceClient:

    def __init__(self, server_address, args):
        self.server_address = server_address
        self.client_rank = args.rank
        self.num_clients = args.world_size
        self.ctx_file = args.config

        context_bytes = open(self.ctx_file, "rb").read()
        self.ctx = ts.context_from(context_bytes)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(self.server_address, options=self.options)
        self.stub = tenseal_allreduce_data_pb2_grpc.AllReduceServiceStub(channel)
        self.data_usage_records = []
        self.comm_time_l = []
        self.index = 0

    def __sum_enc(self, plain_vector):
        # print(">>> client sum encrypted start")

        # encrypt
        encrypt_start = time.time()
        enc_vector = ts.ckks_vector(self.ctx, plain_vector)
        encrypt_time = time.time() - encrypt_start

        # create request
        request_start = time.time()
        enc_vector_bytes = enc_vector.serialize()

        # request_size = len(enc_vector_bytes)

        # send size of msg{ sys.getsizeof(enc_vector_bytes)}

        # print("size of msg: {} bytes".format(sys.getsizeof(enc_vector_bytes)))
        request = tenseal_allreduce_data_pb2.client_msg(
            client_rank=self.client_rank,
            msg=enc_vector_bytes
        )
        request_time = time.time() - request_start

        # comm with server
        comm_start = time.time()
        # print("start comm with server, time = {}".format(time.asctime(time.localtime(time.time()))))
        response = self.stub.sum_enc(request)
        comm_time = time.time() - comm_start
        self.comm_time_l.append(comm_time)

        # deserialize summed enc vector from response
        deserialize_start = time.time()
        assert self.client_rank == response.client_rank
        summed_enc_vector = ts.ckks_vector_from(self.ctx, response.msg)
        deserialize_time = time.time() - deserialize_start

        # received size of msg{ sys.getsizeof(response.msg)}

        # decrypt vector
        decrypt_start = time.time()
        dec_vector = summed_enc_vector.decrypt()
        # print("size of received vector: {}".format(len(dec_vector)))
        decrypt_time = time.time() - decrypt_start

        np_dec_vector =np.array(dec_vector)

        print(">>> client sum enc vector end, cost {:.2f} s: encryption {:.2f} s, create request {:.2f} s, "
              "comm with server {:.2f} s, deserialize {:.2f} s, sent bytes: {}, received bytes: {}"
              .format(time.time() - encrypt_start, encrypt_time, request_time,
                      comm_time, deserialize_time, sys.getsizeof(enc_vector_bytes), sys.getsizeof(response.msg)))

        data_record = {
            "sent_size": sys.getsizeof(enc_vector_bytes),
            "received_size": sys.getsizeof(response.msg)
        }
        self.data_usage_records.append(data_record)

        # print(self.data_usage_records)

        return np_dec_vector

    def transmit(self, plain_vector, operator="sum"):
        trans_start = time.time()
        response = self.__sum_enc(plain_vector) if operator == "sum" else None
        print(">>> client transmission cost {:.2f} s"
              .format(time.time() - trans_start))
        return response
