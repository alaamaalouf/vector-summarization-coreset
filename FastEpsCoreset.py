import numpy as np
import FrankWolfeCoreset as FWC
import copy
import time
import Utils
import functools
import operator
from scipy.stats import ortho_group

N = None
largest_N = None


def fastEpsCoresetOLD(P, w, eps, all_idxs):
    global N, largest_N
    k = np.ceil(2 * np.log2(P.shape[0]) / eps)
    N = P.shape[0]
    if largest_N is None:
        largest_N = N


    if w.ndim < 2:
        w = w[:, np.newaxis]

    if P.shape[0] <= k:
        if P.shape[0] <= np.ceil(8/eps):
            idxs = np.arange(all_idxs.shape[0])
        else:
            _, weights = FWC.FrankWolfeCoreset(P, w, eps).computeCoreset()
            idxs = np.where(weights>0)[0]
        return P[idxs],weights[idxs], all_idxs[idxs]#P[np.where(all_idxs[idxs] < largest_N)[0], :], w[np.where(all_idxs[idxs] < largest_N)[0], 0], all_idxs[np.where(all_idxs[idxs] < largest_N)[0]]

    n,d = P.shape


    k = int(2 * np.log(P.shape[0]) / eps)
    idxs = np.arange(P.shape[0])
    np.random.shuffle(idxs)
    k_clusters = np.array_split(copy.deepcopy(idxs), k)

    idxarray = np.arange(P.shape[0])

    p_groups = P.reshape(current_m, chunk_size, P.shape[1])
    w_groups = w.reshape(current_m, chunk_size)
    idx_group = idxarray.reshape(current_m, chunk_size)
    w_nonzero = np.count_nonzero(w)

    groups_means = np.einsum('ijk,ij->ik', p_groups, w_groups)
    # group_weigts = np.sum(w_groups, axis=1)[:, np.newaxis]
    group_weigts = (np.ones(groups_means.shape[0], dtype=np.float) * 1 / current_m)[:, np.newaxis]
    _, weights = FWC.FrankWolfeCoreset(groups_means, group_weigts, eps/np.log(n)).computeCoreset()

    idxs = np.where(weights > 0)[0]

    C = p_groups[idxs].reshape(-1, d)
    u = (current_m * w_groups[idxs] * weights[idxs]).reshape(-1, 1)



    assert(C.shape[0] == u.shape[0], 'list of lists to list conversion has fault in it!')

    new_idx_array = idx_group[idxs].reshape(-1, 1).flatten()
    new_idx_array = all_idxs[new_idx_array[np.where(new_idx_array < N)[0]]]

    return fastEpsCoreset(copy.deepcopy(C), copy.deepcopy(u), eps, new_idx_array)


def optimalityCondition(n, k, eps):
    if n > k:
        return -1
    elif n > np.ceil(1 / eps):
        return 0
    else:
        return 1


def fastEpsCoreset(P, w, eps, dtype=np.float):
    n, d = P.shape
    k = np.ceil(2 * np.log2(n) / eps)
    Q_P = copy.deepcopy(P)
    Q_w = copy.deepcopy(w)

    if n < np.ceil(k/2):
        return P, w

    chunk_size = int(np.round(np.ceil(n / k)))
    current_m = int(np.ceil(n / chunk_size))

    add_z = chunk_size - int(n % chunk_size)
    w = w.reshape(-1, 1)
    if add_z != chunk_size:
        zeros = np.zeros((add_z, P.shape[1]), dtype=dtype)
        P = np.concatenate((P, zeros))
        zeros = np.zeros((add_z, w.shape[1]), dtype=dtype)
        w = np.concatenate((w, zeros))

    idxarray = np.array(range(P.shape[0]))
    p_groups = P.reshape(current_m, chunk_size, P.shape[1])
    w_groups = w.reshape(current_m, chunk_size)
    idx_group = idxarray.reshape(current_m, chunk_size)
    w_nonzero = np.count_nonzero(w)
    counter = 1
    OLD_P = P
    OLD_W = w
    while w_nonzero > np.ceil(k/2):
            counter += 1
            groups_means = np.einsum('ijk,ij->ik', p_groups, w_groups)
            group_weigts = np.ones(groups_means.shape[0], dtype=dtype) * 1 / current_m
            group_weigts = np.sum(w_groups, axis=1)

            c = np.divide(1, group_weigts, out=np.zeros_like(group_weigts), where=group_weigts != 0)
            groups_means = np.multiply(groups_means, c[:, np.newaxis])
            Cara_p, Cara_w_idx = FWC.FrankWolfeCoreset(groups_means, group_weigts[:, np.newaxis], eps/np.log2(n)).computeCoreset()

            # assert(abs(np.sum(Cara_w_idx) - 1) <= 1e-11, 'Bugzy')
            # mean_before_FW = np.average(groups_means, weights=group_weigts, axis=0)
            # mean_after_FW = np.average(Cara_p, weights=Cara_w_idx.flatten(), axis=0)

            # print('Difference: {}'.format(np.linalg.norm(mean_after_FW - mean_before_FW)))

            IDX = np.nonzero(Cara_w_idx)[0]

            new_P = p_groups[IDX].reshape(-1, d)

            # new_w = (current_m * w_groups[IDX] * Cara_w_idx[IDX]).reshape(-1, 1)

            c = np.divide(Cara_w_idx[IDX], group_weigts[IDX, np.newaxis],
                          out=np.zeros_like(Cara_w_idx[IDX]), where=group_weigts[IDX, np.newaxis] != 0)

            new_w = (w_groups[IDX] * c).reshape(-1, 1)
            # print('difference between two consecutive means: {}'.format(np.linalg.norm(np.average(OLD_P, weights=OLD_W.flatten(), axis=0) - np.average(new_P, weights=new_w.flatten(), axis=0))))

            OLD_P = new_P
            OLD_W = new_w

            new_idx_array = idx_group[IDX].reshape(-1, 1)
            ##############################################################################3
            w_nonzero = np.count_nonzero(new_w)


            chunk_size = int(np.ceil(new_P.shape[0] / k))
            current_m = int(np.ceil(new_P.shape[0] / chunk_size))

            add_z = chunk_size - int(new_P.shape[0] % chunk_size)
            if add_z != chunk_size:
                new_P = np.concatenate((new_P, np.zeros((add_z, new_P.shape[1]), dtype=dtype)))
                new_w = np.concatenate((new_w, np.zeros((add_z, new_w.shape[1]), dtype=dtype)))
                new_idx_array = np.concatenate((new_idx_array, np.zeros((add_z, new_idx_array.shape[1]), dtype=dtype)))
            p_groups = new_P.reshape(current_m, chunk_size, new_P.shape[1])
            w_groups = new_w.reshape(current_m, chunk_size)
            idx_group = new_idx_array.reshape(current_m, chunk_size)
            ###########################################################
    if True:
        # old_mean = np.average(new_P, weights=new_w.flatten(), axis=0)
        # print('Difference between mean of all data with mean before last compression: {}'.format(
        #     np.linalg.norm(np.average(Q_P, weights=Q_w.flatten(), axis=0) - old_mean)))
        # assert(abs(np.sum(new_w) - 1) <= 1e-11, 'sum of weights must be 1!')
        _, weights = FWC.FrankWolfeCoreset(new_P, new_w, eps/1.5).computeCoreset()
        # assert (abs(np.sum(weights) - 1) <= 1e-11, 'sum of weights after compression must be 1!')
        # new_mean = np.average(new_P, weights=weights.flatten(), axis=0)
        # print('Differnce between last means: {}'.format(np.linalg.norm(old_mean - new_mean)))
        idxs = np.where(weights > 0)[0]
        idxs = idxs[np.where(new_idx_array[idxs] < n)[0]].flatten()
        return new_P[idxs], weights[idxs], new_idx_array[idxs].astype(int)

    idxs = np.where(new_w > 0)[0]
    idx = idxs[np.where(new_idx_array.reshape(-1)[idxs] < n)[0]].flatten()
    return new_P[idx], new_w[idx], new_idx_array.reshape(-1)[idx].astype(int).flatten()


def svdCoreset(P, w, k, eps):
    U, D, _ = np.linalg.svd(P, full_matrices=False)
    D_k = D[k:]

    U[k:, k:] = np.multiply(U[k:, k:], D_k[np.newaxis, :]) / np.linalg.norm(D_k, ord=2)
    V = np.hstack((U, np.ones((U.shape[0], 1)))) if False else U
    Q = np.einsum('ij...,i...->ij...', V, V).flatten().reshape(V.shape[0], V.shape[1] ** 2)
    S, u, _, idxs = sparseEpsCoreset(Q, w, (eps / (5 * k)) ** 2 / 16, faster=False)

    W = np.zeros((U.shape[0], ))
    W[idxs.flatten().astype(np.int)] = u.flatten()

    X = ortho_group.rvs(U.shape[1])[:, :k]

    val = np.abs(1 - np.linalg.norm(np.dot(np.multiply(W[:, np.newaxis], P), X)) / np.linalg.norm(np.dot(P, X)))
    # val = np.linalg.norm(np.sum(Q, axis=1) - np.average(S, weights=u.flatten(), axis=0), ord=2)
    assert(val <= eps)
    return u


def sparseEpsCoreset(Q, m, eps, faster=True):
    start_time = time.time()

    # Preprocessing step
    sum_of_m = np.linalg.norm(m.flatten(), ord=1)
    w = (m.flatten() / sum_of_m)[:, np.newaxis]
    mu_m = np.sum(np.multiply(w, Q), axis=0) # compute the mean
    sigma_m = np.sqrt(np.sum(np.multiply(w.flatten(), np.sum((Q - mu_m[np.newaxis, :]) ** 2, axis=1))))  # comute the standard deviation
    P = (Q - mu_m[np.newaxis, :]) / sigma_m

    # shifting the points
    P_prime = np.hstack((P, np.ones((P.shape[0], 1))))
    row_norms = np.expand_dims(np.linalg.norm(P_prime, ord=2, axis=1) ** 2, 1)
    P_prime = np.multiply(P_prime, 1/row_norms)
    w_prime = np.multiply(w, row_norms) / 2
    # old_mean = np.average(P_prime, weights=w_prime.flatten(), axis=0)
    if faster:
        S, u, idxs = fastEpsCoreset(P_prime,  w_prime, eps/2)
        # new_mean = np.average(S, weights=u.flatten(), axis=0)
        # print('Diference bwteen means of P_prime Faster: {}'.format(np.linalg.norm(new_mean - old_mean)))
    else:
        _, u = FWC.FrankWolfeCoreset(P_prime, w_prime, eps).computeCoreset()
        # new_mean = np.average(P_prime, weights=u.flatten(), axis=0)
        # print('Diference bwteen means of P_prime using vanila Frank Wolfe: {}'.format(np.linalg.norm(new_mean - old_mean)))

    # S = P if not faster else P[idxs.flatten(), :]
    if faster:
        S = P[idxs.flatten(), :]
        u = np.multiply(u.flatten(), 2 / row_norms.flatten()[idxs.flatten()])
    else:
        S = P
        u= np.multiply(u.flatten(), 2 / row_norms.flatten())

    # u = np.multiply(u.flatten(), 2 / (row_norms.flatten() if not faster else row_norms.flatten()[idxs.flatten()]))
    return S, u * sum_of_m, time.time() - start_time, idxs


#     assert (np.linalg.norm(np.sum(np.multiply(P_prime, np.multiply(w_prime - x_k, 2 / self.row_norms).T), axis=1))
#         <= 2 * self.epsilon and (np.abs(np.sum(np.multiply(self.w - x_k, 2 / self.row_norms)))
#                                  <= 2 * self.epsilon), 'There is a crucial bug!')


if __name__ == '__main__':
    P = np.random.rand(1000000, 4)
    w = np.ones((P.shape[0], 1))
    # S, u , time_taken = sparseEpsCoreset(P, w, 1/5, True)
    svdCoreset(P, w, 2, 0.9)
    # print('S computed in {:.4f}'.format(time_taken))


# def weakFrankWolfeAlgorithm(P, w, eps):
#     row_norms = np.linalg.norm(P, ord=2, axis=1)[:, np.newaxis]
#     Q = np.divide(P, row_norms)
#     if w.ndim < 2:
#         w = w[:, np.newaxis]
#     w_prime = np.multiply(w, row_norms) / np.sum(np.multiply(row_norms, w))
#
#     _, u = FWC.FrankWolfeCoreset(Q, w_prime, eps).computeCoreset()
#     u = np.divide(u.flatten(), row_norms.flatten()) * np.sum(np.multiply(row_norms, w))
#     return u