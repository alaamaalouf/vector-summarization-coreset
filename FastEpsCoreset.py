import numpy as np
import FrankWolfeCoreset as FWC
import copy
import time


def fastEpsCoreset(P, w, eps):
    if P.shape[0] <= 1 / eps:
        return P, w

    k = int(2 * np.log(P.shape[0]) / eps)
    idxs = np.arange(P.shape[0])
    np.random.shuffle(idxs)
    k_clusters = np.split(copy.deepcopy(idxs), k)

    centers = np.empty(k, P.shape[1])
    sum_w = np.empty(k, )
    for i in range(k):
        centers[i, :] = np.average(a=P[idxs[i], :], axis=1, weights=w[idxs[i]])
        sum_w[i] = np.sum(w[idxs[i]])

    chosen_centers, weights = FWC.FrankWolfeCoreset(centers, sum_w, eps).computeCoreset()

    idxs = np.where(weights > 0)[0]
    C = np.array([P[k_clusters[i], :] for i in idxs])
    u = np.array([[weights[i] * w[j] / sum_w[i] for j in k_clusters[i]] for i in idxs])
    return fastEpsCoreset(copy.deepcopy(C), copy.deepcopy(u), eps)


def sparseEpsCoreset(P, w, eps, faster=True):
    start_time = time.time()
    P_prime = np.hstack((P, np.ones((P.shape[0], 1))))
    row_norms = np.expand_dims(np.linalg.norm(P_prime, ord=2, axis=1) ** 2, 1)
    P_prime = np.multiply(P_prime, 1/row_norms).T
    w_prime = np.multiply(w, row_norms) / 2
    if faster:
        S, u = fastEpsCoreset(P_prime, w_prime, eps)
    else:
        S, u = FWC.FrankWolfeCoreset(P_prime, w_prime, eps).computeCoreset()

    u = np.multiply(u, 2 / row_norms)
    return S, u, time.time() - start_time


#     assert (np.linalg.norm(np.sum(np.multiply(P_prime, np.multiply(w_prime - x_k, 2 / self.row_norms).T), axis=1))
#         <= 2 * self.epsilon and (np.abs(np.sum(np.multiply(self.w - x_k, 2 / self.row_norms)))
#                                  <= 2 * self.epsilon), 'There is a crucial bug!')





