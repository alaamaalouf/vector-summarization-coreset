import numpy as np
import FrankWolfeCoreset as FWC
import copy
import time
import Utils
import functools
import operator


def fastEpsCoreset(P, w, eps):
    if P.shape[0] <= 1 / eps:
        return P, w

    k = int(2 * np.log(P.shape[0]) / eps)
    idxs = np.arange(P.shape[0])
    np.random.shuffle(idxs)
    k_clusters = np.array_split(copy.deepcopy(idxs), k)

    centers = np.empty((k, P.shape[1] - 1))
    sum_w = np.empty(k, )
    for i in range(k):
        centers[i, :] = np.average(a=P[k_clusters[i], :-1], axis=0, weights=w[k_clusters[i]])
        sum_w[i] = np.sum(w[k_clusters[i]])

    _, weights = FWC.FrankWolfeCoreset(centers, sum_w[:, np.newaxis], eps).computeCoreset()

    idxs = np.where(weights > 0)[0]

    C = np.array(functools.reduce(operator.iconcat, [P[k_clusters[i], :] for i in idxs], []))
    u = np.array(functools.reduce(operator.iconcat,
                                  [[weights[i] * w[j] / sum_w[i] for j in k_clusters[i]] for i in idxs], []))
    assert(C.shape[0] == u.shape[0], 'list of lists to list conversion has fault in it!')
    return fastEpsCoreset(copy.deepcopy(C), copy.deepcopy(u), eps)


def sparseEpsCoreset(P, w, eps, faster=True):
    start_time = time.time()
    if w.ndim < 2:
        w = w[:, np.newaxis]
    P_prime = np.hstack((P, np.ones((P.shape[0], 1))))
    row_norms = np.expand_dims(np.linalg.norm(P_prime, ord=2, axis=1) ** 2, 1)
    P_prime = np.multiply(P_prime, 1/row_norms)
    w_prime = np.multiply(w, row_norms) / 2

    if faster:
        S, u = fastEpsCoreset(np.hstack((P_prime, np.arange(P_prime.shape[0], dtype=np.int)[:, np.newaxis])),
                              w_prime.flatten(), eps)
    else:
        S, u = FWC.FrankWolfeCoreset(P_prime, w_prime, eps).computeCoreset()

    u = np.multiply(u, 2 / row_norms[S[:, -1].astype(np.int)]).flatten()
    return P if not faster else P[S[:, -1].astype(np.int), :], u, time.time() - start_time


#     assert (np.linalg.norm(np.sum(np.multiply(P_prime, np.multiply(w_prime - x_k, 2 / self.row_norms).T), axis=1))
#         <= 2 * self.epsilon and (np.abs(np.sum(np.multiply(self.w - x_k, 2 / self.row_norms)))
#                                  <= 2 * self.epsilon), 'There is a crucial bug!')


if __name__ == '__main__':
    P = Utils.readDataset('Synthetic.npy')
    w = np.ones((P.shape[0], 1))
    S, u , time_taken = sparseEpsCoreset(P, w, 1.0/20, True)
    print('S computed in {:.4f}'.format(time_taken))


