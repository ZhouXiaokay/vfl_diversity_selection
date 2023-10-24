#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def global_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23457',
        help='URL specifying how to initialize the package.')
    parser.add_argument('--rank',
                        type=int,
                        default=0,
                        help='Rank of the current process.')
    parser.add_argument('--batch_size',
                        default=100,
                        type=int,
                        help='local batch size')
    parser.add_argument('--num_clients',
                        default=4,
                        type=int,
                        help='the num of clients')
    parser.add_argument('--select_clients',
                        default=2,
                        type=int,
                        help='the num of selected clients')
    parser.add_argument('--world_size',
                        type=int,
                        default=4)
    parser.add_argument('--lr_shapley',
                        default=0.01,
                        type=float,
                        help='learning rate')
    parser.add_argument('--seed',
                        default=3,
                        type=int,
                        help='random seed')
    parser.add_argument('--loss_total',
                        default='0.01',
                        type=float,
                        help='convergence condition')
    parser.add_argument('--start_id',
                        default='9',
                        type=int,
                        help='convergence condition')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=1000)
    parser.add_argument('--epoch_total',
                        type=int,
                        default=3)
    parser.add_argument('--device',
                        default='cpu',
                        type=str,
                        help='the device')
    parser.add_argument('--dataset',
                        type=str,
                        default='libsvm-a8a')
    parser.add_argument('--n_bottom_out',
                        type=int,
                        default=5)
    parser.add_argument('--n_classes',
                        type=int,
                        default=2)
    parser.add_argument('--load_flag',
                        type=bool,
                        default=False)
    parser.add_argument('--avg_flag',
                        type=bool,
                        default=False)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--n-test', type=int, default=300)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--config', type=str, default='/home/userdata/zxk/codes/vfl_diversity_selection/'
                                                      'transmission/tenseal_shapley/ts_ckks_tiny.config')
    parser.add_argument('--a_server_address', type=str, default='127.0.0.1:34567')

    args = parser.parse_args()
    return args
