import numpy as np
from scipy.linalg import lstsq
import random
from sklearn import linear_model
import matplotlib.pyplot as plt
import sys
# from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import time
import math
import scipy.linalg as spla
import itertools
import multiprocessing


# random.seed(7)
def leastsq(X, Y):
    """ Solves the problem Y = XB """
    inv = np.linalg.pinv(np.dot(X.T, X))
    cross = np.dot(inv, X.T)
    beta = np.dot(cross, Y)
    return beta



def CaraIdxCoreset(P, u, dtype='float64'):
    while 1:
        n = np.count_nonzero(u)
        d = P.shape[1]
        u_non_zero = np.nonzero(u)
        if n <= d + 1: return P, u
        A = P[u_non_zero];
        reduced_vec = np.outer(A[0], np.ones(A.shape[0] - 1, dtype=dtype))
        A = A[1:].T - reduced_vec

        idx_of_try = 0;
        const = 10000;
        diff = np.infty;
        cond = sys.float_info.min

        _, _, V = np.linalg.svd(A, full_matrices=True)
        v = V[-1]
        diff = np.max(np.abs(np.dot(A, v)))
        v = np.insert(v, [0], -1 * np.sum(v))

        idx_good_alpha = np.nonzero(v > 0)
        alpha = np.min(u[u_non_zero][idx_good_alpha] / v[idx_good_alpha])

        w = np.zeros(u.shape[0], dtype=dtype)
        tmp = u[u_non_zero] - alpha * v
        tmp[np.argmin(tmp)] = 0.0
        w[u_non_zero] = tmp
        w[u_non_zero][np.argmin(w[u_non_zero])] = 0
        u = w

    return CaraIdxCoreset(P, w)


def updated_cara(P, w, coreset_size, dtype='float64'):
    start_time = time.time()
    d = P.shape[1];
    n = P.shape[0];
    m = 2 * d + 2;  # print (coreset_size,dtype)
    if n <= d + 1: return (P, w, np.array(list(range(0, P.shape[0]))))
    wconst = 1
    w_sum = np.sum(w)
    w = wconst * w / w_sum
    chunk_size = math.ceil(n / m)
    current_m = math.ceil(n / chunk_size)

    add_z = chunk_size - int(n % chunk_size)
    w = w.reshape(-1, 1)
    f = time.time()
    if add_z != chunk_size:
        zeros = np.zeros((add_z, P.shape[1]), dtype=dtype)
        P = np.concatenate((P, zeros))
        f3 = time.time();
        zeros = np.zeros((add_z, w.shape[1]), dtype=dtype)
        w = np.concatenate((w, zeros))

    idxarray = np.array(range(P.shape[0]))

    p_groups = P.reshape(current_m, chunk_size, P.shape[1])
    w_groups = w.reshape(current_m, chunk_size)
    idx_group = idxarray.reshape(current_m, chunk_size)
    w_nonzero = np.count_nonzero(w);
    counter = 1;  # print (w_nonzero, w)

    if not coreset_size: coreset_size = d + 1
    while w_nonzero > coreset_size:
        s0 = time.time()
        counter += 1
        groups_means = np.einsum('ijk,ij->ik', p_groups, w_groups)
        group_weigts = np.ones(groups_means.shape[0], dtype=dtype) * 1 / current_m

        Cara_p, Cara_w_idx = CaraIdxCoreset(groups_means, group_weigts, dtype=dtype)

        IDX = np.nonzero(Cara_w_idx)

        new_P = p_groups[IDX].reshape(-1, d)

        new_w = (current_m * w_groups[IDX] * Cara_w_idx[IDX][:, np.newaxis]).reshape(-1, 1)
        new_idx_array = idx_group[IDX].reshape(-1, 1)
        ##############################################################################3
        w_nonzero = np.count_nonzero(new_w)
        chunk_size = math.ceil(new_P.shape[0] / m)
        current_m = math.ceil(new_P.shape[0] / chunk_size)

        add_z = chunk_size - int(new_P.shape[0] % chunk_size)
        if add_z != chunk_size:
            new_P = np.concatenate((new_P, np.zeros((add_z, new_P.shape[1]), dtype=dtype)))
            new_w = np.concatenate((new_w, np.zeros((add_z, new_w.shape[1]), dtype=dtype)))
            new_idx_array = np.concatenate((new_idx_array, np.zeros((add_z, new_idx_array.shape[1]), dtype=dtype)))
        p_groups = new_P.reshape(current_m, chunk_size, new_P.shape[1])
        w_groups = new_w.reshape(current_m, chunk_size)
        idx_group = new_idx_array.reshape(current_m, chunk_size)
        ###########################################################

    return new_P, w_sum * new_w / wconst, time.time() - start_time, new_idx_array.reshape(-1).astype(int)


def check_cara_out(P, w, Cara_p, Cara_w_Idx, groups_means, group_weigts, w_sum, current_m, Cara_S, Cara_w):
    # print (Cara_S.T.dot(Cara_w.T) * w_sum)
    # print (groups_means.T.dot(group_weigts.T)*w_sum)
    Cara_w_Idx = Cara_w_Idx.reshape(-1, 1)
    print(Cara_p.shape, Cara_w_Idx.shape, groups_means.shape, group_weigts.shape)
    #  print ("checker", P.T.dot(w) - Cara_p.T.dot(Cara_w_Idx) *current_m )
    print("checker", groups_means.T.dot(group_weigts.reshape(-1, 1)) - Cara_p.T.dot(Cara_w_Idx));  # input()


def linregcoreset(P, w, b, c_size=None, is_svd=False, dtype='float64'):
    if not is_svd:
        P_tag = np.append(P, b, axis=1)
    else:
        P_tag = P
    n_tag = P_tag.shape[0];
    d_tag = P_tag.shape[1]
    P_tag = P_tag.reshape(n_tag, d_tag, 1);

    P_tag = np.einsum("ikj,ijk->ijk", P_tag, P_tag)
    P_tag = P_tag.reshape(n_tag, -1);
    n_tag = P_tag.shape[0];
    d_tag = P_tag.shape[1];  # print (w)

    coreset, coreset_weigts, new_idx_array = updated_cara(P_tag.reshape(n_tag, -1), w, c_size, dtype=dtype)

    if coreset is None:     return None, None, None
    coreset_weigts = coreset_weigts[(new_idx_array < P.shape[0])]
    new_idx_array = new_idx_array[(new_idx_array < P.shape[0])]

    # P_tag = np.append(P, b, axis=1)
    # coreset_tag = np.append(P[new_idx_array], b[new_idx_array], axis=1)
    if not is_svd:
        return P[new_idx_array], coreset_weigts.reshape(-1), b[new_idx_array]
    else:
        return P[new_idx_array], coreset_weigts.reshape(-1)


def stream_coreset(P, w, b, splits=None, dtype='float64', big_d=0):
    if splits is None:
        m = int(100000000 / 40);
        lstsq_good_size = 100000000 / 2 - 2 * int(100000000 / 40)
    else:
        m = int(P.shape[0] / splits)
        lstsq_good_size = -1

    d = P.shape[1];
    size_of_coreset = ((d + 1) * (d + 1) + 1)
    if big_d: size_of_coreset = (d + 1) * (d + 1)
    bacthes = splits;  # print (w[0:m])
    if big_d:
        cc, wc, bc = LinRegCoresetBigd(P[0:m], w[0:m], b[0:m], dtype=dtype);
    else:
        cc, wc, bc = linregcoreset(P[0:m], w[0:m], b[0:m], dtype=dtype);

    if cc is None: return None, None, None
    if cc.shape[0] < size_of_coreset and splits:
        add_z = size_of_coreset - cc.shape[0];
        zeros = np.zeros((add_z, cc.shape[1]), dtype=dtype);
        cc = np.concatenate((cc, zeros));
        zeros = np.zeros((add_z), dtype=dtype);
        wc = np.concatenate((wc, zeros));
        zeros = np.zeros((add_z, bc.shape[1]), dtype=dtype);
        bc = np.concatenate((bc, zeros));

    for batch in range(1, bacthes):
        # if cc.shape[0]  + P.shape[0] - m*(batch+1)<  lstsq_good_size :return np.concatenate((cc, P[batch*m:] ))  , np.concatenate((wc, w[batch*m:] )) , np.concatenate((bc, b[batch*m:] ))
        if big_d:
            coreset, new_w, new_b = LinRegCoresetBigd(P[batch * m:(batch + 1) * m], w[batch * m:(batch + 1) * m],
                                                      b[batch * m:(batch + 1) * m], dtype=dtype)
        else:
            coreset, new_w, new_b = linregcoreset(P[batch * m:(batch + 1) * m], w[batch * m:(batch + 1) * m],
                                                  b[batch * m:(batch + 1) * m], dtype=dtype);

        if coreset is None: return None, None, None
        if coreset.shape[0] < size_of_coreset and splits:
            add_z = size_of_coreset - coreset.shape[0];
            zeros = np.zeros((add_z, coreset.shape[1]), dtype=dtype);
            coreset = np.concatenate((coreset, zeros));
            zeros = np.zeros((add_z), dtype=dtype);
            new_w = np.concatenate((new_w, zeros));
            zeros = np.zeros((add_z, new_b.shape[1]), dtype=dtype);
            new_b = np.concatenate((new_b, zeros));
        bc = np.concatenate((bc, new_b))
        cc = np.concatenate((cc, coreset))
        wc = np.concatenate((wc, new_w))
    # print (cc.shape,wc.shape, bc.shape)
    return cc, wc, bc


def LinRegCoresetBigd(P, w, b, is_svd=False, dtype='float64'):
    if P.shape[0] < P.shape[1] + 1: return P, w, b
    if not is_svd:
        P_tag = np.append(P, b, axis=1)
    else:
        P_tag = P
    # PTP = np.dot(P_tag.T , P_tag)
    CTC = FastCovEstimationForBigd(P_tag, w)
    # print (P_tag)
    # print (CTC - PTP)
    # C_T =np.linalg.cholesky(PTP)
    C_T = np.linalg.cholesky(CTC)
    C = C_T.T
    if not is_svd:
        return C[:, :-1], np.ones(C.shape[0]), C[:, -1].reshape(-1, 1)
    else:
        return C, np.ones(C.shape[0])


def FastCovEstimationForBigd(P, w):
    global cara_mat_2d

    def cara_mat_2d(pair):
        tz = time.time()
        pair = list(pair)
        A = P[:, pair]
        C, u = linregcoreset(A, w, None, c_size=None, is_svd=True)
        C_tag = C * np.sqrt(u[:, np.newaxis])
        cov = C_tag.T.dot(C_tag)
        tz2 = time.time()
        # print ("look", tz2- tz)
        return cov

    d = P.shape[1]
    CTC = np.zeros((d, d))
    t0 = time.time()
    idx_pairs = list(itertools.combinations(range(d), 2))
    t1 = time.time()
    pool = multiprocessing.Pool()
    results = list(pool.map(cara_mat_2d, idx_pairs))
    pool.close()
    t2 = time.time()
    for key, pair in enumerate(idx_pairs):
        cov = results[key]
        CTC[pair[0], pair[1]], CTC[pair[1], pair[0]] = cov[0, 1], cov[0, 1]
        if CTC[pair[0], pair[0]] == 0: CTC[pair[0], pair[0]] = cov[0, 0]
        if CTC[pair[1], pair[1]] == 0: CTC[pair[1], pair[1]] = cov[1, 1]
    t3 = time.time()
    print(t3 - t2, t2 - t1, t1 - t0)
    return CTC

# Pa =  np.floor(np.random.rand(1000,5)*100)
# w = np.ones(1000);
# print (FastCovEstimationForBigd(Pa,w))
# print (Pa.T.dot(Pa))