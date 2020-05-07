import numpy as np
import cvxpy as cp


class FrankWolfeCoreset(object):
    def __init__(self, P, w, epsilon):
        self.P = P
        self.n = P.shape[0]
        self.w = w
        self.Q = None
        self.prepareData()
        self.epsilon = epsilon
        self.T = int(np.ceil(1.0 / epsilon))

    def updatedData(self, P, w=None, epsilon=None):
        if epsilon is not None:
            self.epsilon = epsilon

        self.P = P
        if w is not None:
            self.w = w

        self.prepareData()

    def computeCoreset(self):
        return self.applyFrankWolfe()
        # u = np.multiply(u, 2 / self.row_norms)
        # return self.P[np.where(u > 0)[0], :], u

    def applyFrankWolfe(self):
        e_1 = np.eye(self.n, 1)
        a = cp.Variable(1, name='a')
        a.value = [0.0]
        term = (lambda j, x:
                cp.sum(cp.multiply(self.Q,
                                   (np.tile(self.w - x, (1, 3)) - a * np.tile(np.roll(e_1, j) - x, (1, 3))).T),
                       axis= 1))
        # term = (lambda j, x: cp.sum(cp.matmul(self.Q, cp.diag(self.w - x - a * (np.roll(e_1, j) - x))), axis=1))
        grad_func = (lambda x: np.dot(np.sum(np.multiply(self.Q, (self.w - x).T),
                                             axis=1)[:, np.newaxis].T, self.Q).flatten())

        vals = np.empty(self.n, )
        for i in range(self.n):
            x_k = np.roll(e_1, i)
            vals[i] = -cp.matmul(term(0, x_k).T, term(0, x_k)).value

        j = np.argmax(vals)
        x_k = np.roll(e_1, j)

        for i in range(self.T):
            j = np.argmax(grad_func(x_k))
            objective_func = -cp.norm(term(j, x_k)) ** 2
            prob = cp.Problem(cp.Maximize(objective_func), [a >= 0, a <= 1])
            prob.solve()

            x_k = x_k + a.value * (np.roll(e_1, j) - x_k)

        return self.P, x_k

    @staticmethod
    def main():
        P = np.random.randn(998, 2)
        P = np.vstack((P, 10000 * np.random.rand(2, 2)))
        w = np.ones((1000, 1)) / 1000
        P = P - np.mean(P, 0)
        P = P / np.sqrt(np.sum(np.multiply(w, np.sum(P ** 2, axis=1))))
        frank_wolfe = FrankWolfeCoreset(P, w, 0.01)
        S, u = frank_wolfe.computeCoreset()
        print('Our coreset is: {}'.format(S))


if __name__ == '__main__':
    FrankWolfeCoreset.main()