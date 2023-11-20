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


def load_credit_data(csv_path='/home/userdata/zxk/data/credit/credit.csv'):
    data = load_csv(csv_path)
    data.rename(columns={'default.payment.next.month': 'def_pay'}, inplace=True)
    data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
    x = data.drop(['def_pay', 'ID'], axis=1)
    x_std = StandardScaler().fit_transform(x)
    y = data.def_pay
    # dataset = {'id': data.index.values, 'x': x.values, 'y': y.values}
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_bank_data(csv_path='/home/userdata/zxk/data/bank/bank.csv'):
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


def load_mushroom_data(csv_path='/home/userdata/zxk/data/mushroom/mushrooms.csv'):
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


def load_covtype_data(data_path='/home/userdata/zxk/data/covtype/covtype.libsvm.binary.scale.bz2'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1] - 1
    x_std = StandardScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_g2_data(data_path='/home/userdata/zxk/data/g2/g2-128-10.txt'):
    x = load_txt(data_path)
    c_1 = np.zeros(1024)
    c_2 = np.ones(1024)
    y = np.concatenate((c_1, c_2))
    idx = np.arange(0, x.shape[0])
    dataset = {'id': idx, 'x': x, 'y': y}
    return dataset


def load_phishing_data(data_path='/home/userdata/zxk/data/phishing/phishing.txt'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_adult_data(data_path='/home/userdata/zxk/data/libsvm'):
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


def load_web_data(data_path='/home/userdata/zxk/data/libsvm'):
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


def load_ijcnn_data(data_path='/home/userdata/zxk/data/libsvm'):
    train_path = data_path + '/ijcnn1.tr'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/ijcnn1.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_splice_data(data_path='/home/userdata/zxk/data/libsvm/'):
    train_path = data_path + '/splice'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/splice.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_madelon_data(data_path='/home/userdata/zxk/data/madelon/'):
    train_path = data_path + '/madelon'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/madelon.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_gisette_data(data_path='/home/userdata/zxk/data/gisette/'):
    train_path = data_path + '/gisette_scale'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/gisette_scale.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_SUSY_data(data_path='/home/userdata/zxk/data/SUSY/SUSY'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_heart_data(data_path='/home/userdata/zxk/data/heart/heart'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_spambase_data(csv_path="/home/userdata/zxk/data/spambase/spambase.csv"):
    data = load_csv(csv_path)

    le = LabelEncoder()
    y = le.fit_transform(data.iloc[:, -1])

    x = data.iloc[:, :-1]
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y}

    return dataset


def load_HIGGS_data(data_path='/home/userdata/zxk/data/HIGGS/HIGGS', sample_size=None, random_seed=None):
    data = load_svmlight_file(data_path)

    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0

    # 分别获取两个类别的索引
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # 计算每个类别的采样数量
    if sample_size is not None:
        if random_seed is not None:
            np.random.seed(random_seed)

        num_samples_class_0 = int(sample_size / 2)
        num_samples_class_1 = sample_size - num_samples_class_0

        # 随机选择每个类别的索引
        sampled_indices_class_0 = np.random.choice(class_0_indices, num_samples_class_0, replace=False)
        sampled_indices_class_1 = np.random.choice(class_1_indices, num_samples_class_1, replace=False)

        # 合并两个类别的采样索引
        sampled_indices = np.concatenate([sampled_indices_class_0, sampled_indices_class_1])

        # 使用采样索引来选择数据
        x = x[sampled_indices]
        y = y[sampled_indices]

    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_realsim_data(data_path='/home/userdata/zxk/data/real-sim/real-sim', sample_size=None, random_seed=None):
    data = load_svmlight_file(data_path)

    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0

    # 分别获取两个类别的索引
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # 计算每个类别的采样数量
    if sample_size is not None:
        if random_seed is not None:
            np.random.seed(random_seed)

        num_samples_class_0 = int(sample_size / 2)
        num_samples_class_1 = sample_size - num_samples_class_0

        # 随机选择每个类别的索引
        sampled_indices_class_0 = np.random.choice(class_0_indices, num_samples_class_0, replace=False)
        sampled_indices_class_1 = np.random.choice(class_1_indices, num_samples_class_1, replace=False)

        # 合并两个类别的采样索引
        sampled_indices = np.concatenate([sampled_indices_class_0, sampled_indices_class_1])

        # 使用采样索引来选择数据
        x = x[sampled_indices]
        y = y[sampled_indices]

    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_epsilon_data(data_path='/home/userdata/zxk/data/epsilon/', sample_size=None, random_seed=None):
    train_path = data_path + '/epsilon_normalized'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/epsilon_normalized.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    # 分别获取两个类别的索引
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # 计算每个类别的采样数量
    if sample_size is not None:
        if random_seed is not None:
            np.random.seed(random_seed)

        num_samples_class_0 = int(sample_size / 2)
        num_samples_class_1 = sample_size - num_samples_class_0

        # 随机选择每个类别的索引
        sampled_indices_class_0 = np.random.choice(class_0_indices, num_samples_class_0, replace=False)
        sampled_indices_class_1 = np.random.choice(class_1_indices, num_samples_class_1, replace=False)

        # 合并两个类别的采样索引
        sampled_indices = np.concatenate([sampled_indices_class_0, sampled_indices_class_1])

        # 使用采样索引来选择数据
        x = x[sampled_indices]
        y = y[sampled_indices]

    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_magicGammaTelescope_data(csv_path='/home/userdata/zxk/data/magic+gamma+telescope/magic04.csv'):
    data = load_csv(csv_path)

    le = LabelEncoder()
    y = le.fit_transform(data.iloc[:, -1])
    y[y == 'g'] = 0
    y[y == 'h'] = 1

    x = data.iloc[:, :-1]
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y}

    return dataset


def load_smkdrk_data(csv_path='/home/userdata/zxk/data/SMK_DRK/smoking_driking_dataset_Ver01.csv'):
    data = load_csv(csv_path)

    data['sex'] = LabelEncoder().fit_transform(data['sex'])
    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['DRK_YN']
    x = data.copy()
    x.drop('DRK_YN', axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_nba_player_data(csv_path='/home/userdata/zxk/data/nba_players/nba-players.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['target_5yrs']
    x = data.copy()
    columns_to_drop = ['id', 'name', 'target_5yrs']
    x.drop(columns_to_drop, axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_heart_disease_data(csv_path='/home/userdata/zxk/data/heart_disease/heart_disease_health_indicators.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['HeartDiseaseorAttack']
    x = data.copy()
    x.drop('HeartDiseaseorAttack', axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_rice_data(csv_path='/home/userdata/zxk/data/rice/riceClassification.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['Class']
    x = data.copy()
    x.drop(['id', 'Class'], axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_creditcard_data(csv_path='/home/userdata/zxk/data/creditcard/creditcard.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['class']
    x = data.copy()
    x.drop(['time', 'amount', 'class'], axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_smoke_data(csv_path='/home/userdata/zxk/data/smoke/smoke_detection_iot.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['Fire Alarm']
    x = data.copy()
    x.drop(['Fire Alarm', 'ID'], axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_dummy_partition_with_label(dataset, num_clients, client_rank):
    split_x = vertical_split(dataset, num_clients)
    x = split_x[client_rank]
    y = dataset['y']

    return x, y


def choose_dataset(d_name):
    if d_name == 'credit':
        data = load_credit_data()
    elif d_name == 'bank':
        data = load_bank_data()
    elif d_name == 'mushroom':
        data = load_mushroom_data()
    elif d_name == 'covtype':
        data = load_covtype_data()
    elif d_name == 'adult':
        data = load_adult_data()
    elif d_name == 'web':
        data = load_web_data()
    elif d_name == 'phishing':
        data = load_phishing_data()
    elif d_name == 'ijcnn':
        data = load_ijcnn_data()
    elif d_name == 'splice':
        data = load_splice_data()
    elif d_name == 'SUSY':
        data = load_SUSY_data()
    elif d_name == 'heart':
        data = load_heart_data()
    elif d_name == 'HIGGS':
        data = load_HIGGS_data('/home/userdata/zxk/data/HIGGS/HIGGS', 100000, 1)
    elif d_name == 'madelon':
        data = load_madelon_data()
    elif d_name == 'real-sim':
        data = load_realsim_data('/home/userdata/zxk/data/real-sim/real-sim', 100000, 1)
    elif d_name == 'epsilon':
        data = load_epsilon_data('/home/userdata/zxk/data/epsilon', 10000, 1)
    elif d_name == 'gisette':
        data = load_gisette_data()
    elif d_name == 'spambase':
        data = load_spambase_data()
    elif d_name == 'magicGammaTelescope':
        data = load_magicGammaTelescope_data()
    elif d_name == 'smk-drk':
        data = load_smkdrk_data()
    elif d_name == 'heart-disease':
        data = load_heart_disease_data()
    elif d_name == 'rice':
        data =load_rice_data()
    elif d_name == 'creditcard':
        data = load_creditcard_data()
    elif d_name == 'smoke':
        data = load_smoke_data()
    else:
        print("there's not this dataset")
        return -1
    return data