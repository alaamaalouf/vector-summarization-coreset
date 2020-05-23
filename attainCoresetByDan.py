import numpy as np
import scipy as sp
from scipy.linalg import null_space
import FrankWolfeCoreset as FWC
import copy
import time
from FastEpsCoreset import sparseEpsCoreset

SMALL_NUMBER = 1e-7


def checkIfPointsAreBetween(p, start_line, end_line):
    return (np.all(p <= end_line) and np.all(p >= start_line)) or (np.all(p <= start_line) and np.all(p >= end_line))


def checkIfPointOnSegment(p, start_line, end_line, Y):
    if checkIfPointsAreBetween(p, start_line, end_line):
        _, D, V = np.linalg.svd(np.vstack((np.zeros(start_line.shape), end_line - start_line)))
        if np.linalg.norm(np.dot(p - start_line, Y)) < 1e-11:
            return True
    return False


def attainCoresetByDanV2(P, u, eps):
    n, d = P.shape
    j = 0
    c = P[j, :]
    w = np.empty((n,))
    w[j] = 1
    num_iter = 1
    m = 1

    assert (np.all(np.sum(P ** 2, 1) - 1 <= SMALL_NUMBER), 'The data is not properly scaled')

    for iter in range(1, int(1 / eps)):
        num_iter += 1
        D = np.dot(P, c)
        D[D <= SMALL_NUMBER] = 0
        j = np.argmin(D)
        p = P[j, :]

        norm_c = np.linalg.norm(c)
        norm_p = np.linalg.norm(p)
        cp = p.dot(c)
        norm_c_p = np.sqrt(norm_p ** 2 + norm_c ** 2 - 2 * cp)
        assert (np.abs(norm_c_p - np.linalg.norm(c - p)) <= SMALL_NUMBER, 'Bug')

        v = p - c
        c_1 = p - (v / np.linalg.norm(v)) * p.dot(v / np.linalg.norm(v))

        norm_c_1 = np.linalg.norm(c_1)
        alpha = np.linalg.norm(c - c_1) / norm_c_p

        assert (alpha < 1, 'Bug')

        w = w * (1 - np.abs(alpha))
        w[j] += alpha
        w /= np.sum(w)

        c = c_1

    return w


def attainCoresetByDanV1(P, u, eps):
    ts = time.time()
    if u.ndim < 2:
        u = u[:, np.newaxis]
    E_u = np.sum(np.multiply(P, u), axis=0)
    x = np.sum(np.multiply(u.flatten(), np.sqrt(np.sum((P - E_u) ** 2, axis=1))))
    lifted_P = np.hstack((P - E_u, x * np.ones((P.shape[0], 1))))
    v = np.sum(np.multiply(u.flatten(), np.sqrt(np.sum(lifted_P ** 2, axis=1))))
    Q = np.multiply(lifted_P, 1 / np.linalg.norm(lifted_P, ord=2, axis=1)[:, np.newaxis])
    s = np.multiply(u.flatten(), 1 / v * np.linalg.norm(lifted_P, ord=2, axis=1))

    last_entry_vec = np.zeros((1, lifted_P.shape[1]))
    last_entry_vec[0, -1] = x / v

    H = Q - last_entry_vec

    tau = v / int(np.sqrt(1 / eps))
    alpha = 2 * (1 + 2 * (1 + tau ** 2) / (1 - tau) ** 2)

    beta = int(np.ceil(alpha / eps))
    h = np.empty((beta, H.shape[1]))
    c_i = copy.deepcopy(h)
    c_i[0, :] = np.random.choice(np.arange(P.shape[0]))
    origin = np.zeros((H.shape[1],))
    for i in range(beta - 1):
        h[i, :] = H[np.argmax(np.linalg.norm(H - c_i[i, :], ord=2, axis=1)), :]
        _, D, V = np.linalg.svd(np.vstack((np.zeros(h[i, :].shape), h[i, :] - c_i[i, :])))
        orth_line_segment = null_space(V[np.where(D > 1e-11)[0], :])
        project_origin = -np.dot(origin - c_i[i, :], orth_line_segment.dot(orth_line_segment.T))
        if checkIfPointOnSegment(project_origin, c_i[i, :], h[i, :], orth_line_segment):
            c_i[i + 1, :] = project_origin
        else:
            dist1, dist2 = np.linalg.norm(project_origin - c_i[i, :]), np.linalg.norm(project_origin - h[i, :])
            c_i[i + 1, :] = h[i, :] if dist2 < dist1 else c_i[i, :]

    _, w_prime = FWC.FrankWolfeCoreset(Q, s[:, np.newaxis], eps).computeCoreset()

    w_double_prime = np.multiply(v * w_prime.flatten(), 1 / np.linalg.norm(lifted_P, ord=2, axis=1))
    w = w_double_prime / np.sum(w_double_prime)

    S = P[np.where(w > 0)[0], :]

    return S, w[np.where(w > 0)[0]], time.time() - ts, np.where(w > 0)[0]


if __name__ == '__main__':
    n = 600000
    d = 40
    P = np.random.randn(n, d) * 1000
    # P = np.load('Synthetic.npy')
    w = np.ones((n, 1)) / n
    P = P - np.mean(P, 0)
    P = P / np.sqrt(np.sum(np.multiply(w.flatten(), np.sum(P ** 2, axis=1))))
    ts = time.time()
    Z = 1
    eps = 0.1
    for i in range(1):
        S, u, ts = sparseEpsCoreset(P, w, eps, faster=False)
        # assert (abs(np.sum(u) - 1) <= 1e-11, 'Bugzy in SLOW!')
        print('n : {}, u:{}, real_n: {}'.format(S.shape[0], u.shape[0], np.count_nonzero(u.flatten())))

        print('Our done in {:.4f}'.format(ts))
        print('error with our slow : {}'.format(
            np.linalg.norm(np.average(P, weights=w.flatten(), axis=0) - np.average(S, weights=u.flatten(), axis=0))**Z))
        S, u, ts = sparseEpsCoreset(P, w, eps, faster=True)
        # assert (abs(np.sum(u) - 1) <= 1e-11, 'Bugzy in FAST!')
        print('Our fast done in {:.4f}'.format(ts))
        print('n : {}, u:{}, real_n: {}'.format(S.shape[0], u.shape[0], np.count_nonzero(u)))
        print('error with our fast {}'.format(
            np.linalg.norm(np.average(P, weights=w.flatten(), axis=0) - np.average(S, weights=u.flatten(), axis=0))**Z))

        ts = time.time()
        S, u = attainCoresetByDanV1(P, w, eps)
        assert (abs(np.sum(u) - 1) <= 1e-11, 'Bugzy in Dan\'s code!')
        print('Their time is {:.4f}'.format(time.time() - ts))
        print('n : {}, u:{}'.format(S.shape[0], u.shape[0]))

        print('Their error {}'.format(
            np.linalg.norm(np.average(P, weights=w.flatten(), axis=0) - np.average(S, weights=u.flatten(), axis=0))**Z))
