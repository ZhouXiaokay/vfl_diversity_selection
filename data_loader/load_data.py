import numpy as np
import pandas as pd
from utils.helpers import seed_torch
from conf import global_args_parser

global_args = global_args_parser()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.datasets import load_svmlight_file


def load_csv(f_path):
    """ Load data set """
    data = pd.read_csv(f_path)
    return data

def load_txt(txt_path):
    data = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split()
            data.append(splits)
    np_data = np.array(data).astype(int)
    return np_data

def vertical_split(data, num_users):
    seed_torch()
    num_features = int(data['x'].shape[1] / num_users)
    f_id = [i for i in range(data['x'].shape[1])]
    split_result = {}
    split_f = []
    for i in range(num_users):
        if i == num_users - 1:
            leave_id = list(set(f_id) - set(sum(split_f, [])))
            split_f.append(list(leave_id))
        else:
            t = set(np.random.choice(f_id, num_features, replace=False))
            split_f.append(list(t))
            f_id = list(set(f_id) - t)
    client_rank = 0
    for item in split_f:
        x_sub = concat_split_result(item, data['x'])
        split_result[client_rank] = x_sub
        client_rank += 1

    return split_result


def concat_split_result(r_list, npx):
    x_list = []
    for i in r_list:
        x_list.append(npx[:, i:i + 1])
    return np.concatenate(tuple(x_list), axis=1)


def load_credit_data(csv_path='/home/userdata/zxk/codes/vfl_data_valuation/data/credit/credit.csv'):
    data = load_csv(csv_path)
    data.rename(columns={'default.payment.next.month': 'def_pay'}, inplace=True)
    data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
    x = data.drop(['def_pay', 'ID'], axis=1)
    x_std = StandardScaler().fit_transform(x)
    y = data.def_pay
    # dataset = {'id': data.index.values, 'x': x.values, 'y': y.values}
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_bank_data(csv_path='/home/userdata/zxk/codes/vfl_data_valuation/data/bank/bank.csv'):
    data = load_csv(csv_path)
    data = data.drop(['customer_id'], axis=1)
    data['country'] = LabelEncoder().fit_transform(data['country'])
    data['gender'] = LabelEncoder().fit_transform(data['gender'])

    y = data['churn']
    x = data.copy()
    x.drop('churn', axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    # print(x_std)
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_mushroom_data(csv_path='/home/userdata/zxk/codes/vfl_data_valuation/data/mushroom/mushrooms.csv'):
    data = load_csv(csv_path)
    data.drop(columns='veil-type', inplace=True)
    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])
    y = data['class']
    x = data.copy()
    x.drop('class', axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_covtype_data(data_path='/home/userdata/zxk/codes/vfl_data_valuation/data/covtype/covtype.libsvm.binary.scale.bz2'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1] - 1
    x_std = StandardScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_g2_data(data_path='/home/userdata/zxk/codes/vfl_data_valuation/data/g2/g2-128-10.txt'):
    x = load_txt(data_path)
    c_1 = np.zeros(1024)
    c_2 = np.ones(1024)
    y = np.concatenate((c_1, c_2))
    idx = np.arange(0, x.shape[0])
    dataset = {'id': idx, 'x': x, 'y': y}
    return dataset


def load_phishing_data(data_path='/home/userdata/zxk/codes/vfl_data_valuation/data/phishing/phishing.txt'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_adult_data(data_path='/home/userdata/zxk/codes/vfl_data_valuation/data/libsvm'):
    train_path = data_path + '/a8a.txt'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    train_X = train_X[:, :-1]

    test_path = data_path + '/a8a.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = StandardScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_web_data(data_path='/home/userdata/zxk/codes/vfl_data_valuation/data/libsvm'):
    train_path = data_path + '/w8a.txt'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/w8a.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_dummy_partition_with_label(dataset, num_clients, client_rank):
    split_x = vertical_split(dataset, num_clients)
    x = split_x[client_rank]
    y = dataset['y']

    return x, y


def choose_dataset(d_name):
    data = load_credit_data()
    if d_name == 'credit':
        return data
    elif d_name == 'bank':
        data = load_bank_data()
        return data
    elif d_name == 'mushroom':
        data = load_mushroom_data()
        return data
