import numpy as np
from sklearn.metrics import pairwise_distances

a = np.array([[4, 4, 6, 7], [4, 5, 6, 7], [4, 3, 6, 7], [4, 2, 6, 7], [4, 1, 6, 7], [4, 0, 6, 7]])
# b = np.array([[0, 3], [1, 2]])
# print(a[:, b][0, :, :])
#
# # print(np.argmax(a, axis=1))
# print(np.tile(np.array([1, 2]), (2, 1)))
clients_list = [1, 2, 3, 4, 5, 6]


def stochastic_greedy(norm_diff, clients, num_clients, subsample=0.8):
    # initialize the ground set and the selected set
    V_set = set(range(len(clients)))
    SUi = set()

    m = max(num_clients, int(subsample * len(clients)))
    for ni in range(num_clients):
        if m < len(V_set):
            R_set = np.random.choice(list(V_set), m, replace=False)
        else:
            R_set = list(V_set)
        if ni == 0:
            marg_util = norm_diff[:, R_set].sum(0)
            # print("here",marg_util,R_set, norm_diff[:, R_set])
            i = marg_util.argmin()
            client_min = norm_diff[:, R_set[i]]
        else:
            client_min_R = np.minimum(client_min[:, None], norm_diff[:, R_set])
            marg_util = client_min_R.sum(0)
            i = marg_util.argmin()
            client_min = client_min_R[:, i]
        SUi.add(R_set[i])
        V_set.remove(R_set[i])
    return SUi


def lazy_greedy(norm_diff, num_clients):
    # initialize the ground set and the selected set
    V_set = set(range(len(clients)))
    SUi = set()

    S_util = 0
    marg_util = norm_diff.sum(0)
    i = marg_util.argmin()
    L_s0 = 2. * marg_util.max()
    marg_util = L_s0 - marg_util
    client_min = norm_diff[:, i]
    # print(i)
    SUi.add(i)
    V_set.remove(i)
    S_util = marg_util[i]
    marg_util[i] = -1.

    while len(SUi) < num_clients:
        argsort_V = np.argsort(marg_util)[len(SUi):]
        for ni in range(len(argsort_V)):
            i = argsort_V[-ni - 1]
            SUi.add(i)
            client_min_i = np.minimum(client_min, norm_diff[:, i])
            SUi_util = L_s0 - client_min_i.sum()

            marg_util[i] = SUi_util - S_util
            if ni > 0:
                if marg_util[i] < marg_util[pre_i]:
                    if ni == len(argsort_V) - 1 or marg_util[pre_i] >= marg_util[argsort_V[-ni - 2]]:
                        S_util += marg_util[pre_i]
                        # print(pre_i, L_s0 - S_util)
                        SUi.remove(i)
                        SUi.add(pre_i)
                        V_set.remove(pre_i)
                        marg_util[pre_i] = -1.
                        client_min = client_min_pre_i.copy()
                        break
                    else:
                        SUi.remove(i)
                else:
                    if ni == len(argsort_V) - 1 or marg_util[i] >= marg_util[argsort_V[-ni - 2]]:
                        S_util = SUi_util
                        # print(i, L_s0 - S_util)
                        V_set.remove(i)
                        marg_util[i] = -1.
                        client_min = client_min_i.copy()
                        break
                    else:
                        pre_i = i
                        SUi.remove(i)
                        client_min_pre_i = client_min_i.copy()
            else:
                if marg_util[i] >= marg_util[argsort_V[-ni - 2]]:
                    S_util = SUi_util
                    # print(i, L_s0 - S_util)
                    V_set.remove(i)
                    marg_util[i] = -1.
                    client_min = client_min_i.copy()
                    break
                else:
                    pre_i = i
                    SUi.remove(i)
                    client_min_pre_i = client_min_i.copy()
    return SUi


# print(stochastic_greedy(clients_list, 4, 0.5))
clients = [0, 1, 2, 3]
b = np.asarray([[8.09], [7.06], [6.37], [11.56]])
# b = 30.24 - b
b_norm_diff = pairwise_distances(b, metric="euclidean")
# b_norm_diff = np.asarray([[0. ,7.71,6.62,6.44], [1.09 0.   0.18 2.86] ,[1.27 0.18 0.   3.04] ,[1.77 2.86 3.04 0.  ]])
np.fill_diagonal(b_norm_diff, 0)
print(b_norm_diff)
print(stochastic_greedy(b_norm_diff, clients, 2))
print(lazy_greedy(b_norm_diff, 2))

V_set = set(range(len(clients)))
R_set = np.random.choice(list(V_set), 2, replace=False)
print(R_set)
marg_util = b[R_set].sum(1)
print(marg_util)

# np.random.seed(2323)
def stochastic_greedy_test(avg_dist_list, clients, num_clients, subsample=0.5):
    # initialize the ground set and the selected set
    V_set = set(range(len(clients)))
    SUi = set()

    m = max(num_clients, int(subsample * len(clients)))
    for ni in range(num_clients):
        if m < len(V_set):
            R_set = np.random.choice(list(V_set), m, replace=False)
        else:
            R_set = list(V_set)
        if ni == 0:
            marg_util = avg_dist_list[R_set].sum(1)
            # print("here",marg_util,R_set, norm_diff[:, R_set])
            i = marg_util.argmax()
            client_max = avg_dist_list[R_set[i]]
        else:
            client_max_R = np.maximum(client_max[:], avg_dist_list[R_set])
            print("here", client_max_R)
            marg_util = client_max_R.sum(1)
            i = marg_util.argmax()
            client_max = client_max_R[i]
        SUi.add(R_set[i])
        V_set.remove(R_set[i])
    return SUi


def stochastic_greedy_test1(avg_dist_list, clients, num_clients, subsample=0.6):

    # initialize the ground set and the selected set
    V_set = set(range(len(clients)))
    SUi = set()

    m = max(num_clients, int(subsample * len(clients)))
    for ni in range(num_clients):
        if m < len(V_set):
            R_set = np.random.choice(list(V_set), m, replace=False)
        else:
            R_set = list(V_set)
        if ni == 0:
            marg_util = avg_dist_list[R_set].sum(1)
            # print("here",marg_util,R_set, norm_diff[:, R_set])
            i = marg_util.argmax()
            # client_max = avg_dist_list[R_set[i]]
        else:
            # client_max_R = np.maximum(client_max[:], avg_dist_list[R_set])
            # print("here", client_max_R)
            marg_util = avg_dist_list[R_set].sum(1)
            i = marg_util.argmax()
            # client_max = client_max_R[i]
        SUi.add(R_set[i])
        V_set.remove(R_set[i])
    return SUi


print(stochastic_greedy_test(b, clients, 2))
print(stochastic_greedy_test1(b, clients, 2))

shuffle_ind = np.arange(4)
print(b)
np.random.shuffle(shuffle_ind)
print(shuffle_ind)
print(b[shuffle_ind])

# c = np.asarray([[0.309], [0.402], [0.236], [0.178], [0.239], [0.230], [0.154], [0.92]])
# c = 2.67 - c
# clients = [0, 1, 2, 3, 4, 5, 6, 7]
#
# c_norm_diff = pairwise_distances(c, metric="euclidean")
# np.fill_diagonal(c_norm_diff, 0)
# print(c_norm_diff)
# print(stochastic_greedy(c_norm_diff, clients, 4))
# print(lazy_greedy(c_norm_diff, 4))

# factor = [1] * 4
# for i in range(1, 4):
#     factor[i] = factor[i - 1] * i
# print(factor)

# def get_utility_key(client_attendance):
#     key = 0
#     for i in reversed(client_attendance):
#         key = 2 * key + i
#     return key
#
# print(get_utility_key([1,0,0,0]))
# print(8//2)
