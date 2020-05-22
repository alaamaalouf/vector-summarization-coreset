import numpy as np
import time
from scipy.optimize import minimize_scalar



class FrankWolfeCoreset(object):
    def __init__(self, P, w, epsilon):
        self.P = P
        self.n = P.shape[0]
        self.w = w
        self.Q = P.T
        self.epsilon = epsilon
        self.T = int(np.ceil(1.0 / epsilon));self.e_1= np.eye(self.n,1)

    def updatedData(self, P, w=None, epsilon=None):
        if epsilon is not None:
            self.epsilon = epsilon

        self.P = P
        if w is not None:
            self.w = w

    def computeCoreset(self):
        return self.applyFrankWolfe()
        # u = np.multiply(u, 2 / self.row_norms)
        # return self.P[np.where(u > 0)[0], :], u
    

    def term(self, x, P, x_k, j):
        return np.linalg.norm(np.sum(np.einsum('ij,j->ij', self.Q, (self.w-x_k - x * (np.roll(self.e_1, j) - x_k)).flatten()), axis=1)) ** 2
    
    def applyFrankWolfe(self):
        ##se_1 = np.eye(self.n, 1)
    
        #a.value = [0.0]
       # term = (lambda j, x:
       #         cp.sum(
       #             cp.multiply(self.Q, (np.tile(self.w - x, (1, self.Q.shape[0])) -
       #                                  a * np.tile(np.roll(e_1, j) - x, (1, self.Q.shape[0]))).T), axis=1))
        # term = (lambda j, x: cp.sum(cp.matmul(self.Q, cp.diag(self.w - x - a * (np.roll(e_1, j) - x))), axis=1))
        
        grad_func = (lambda x: np.dot(np.sum(np.multiply(self.Q, (self.w - x).T),
                                             axis=1)[:, np.newaxis].T, self.Q).flatten())
        
        vals = np.empty(self.n, )
        # t = time.time()
        mean_diff = np.sum(np.einsum('ij,j->ij', self.Q, self.w.flatten()), axis=1)
        #v = np.sum(self.Q, axis=1)
        
        for i in range(self.n):
            mean_diff -= self.Q[:, i]
            vals[i] = -np.linalg.norm(mean_diff)
            mean_diff += self.Q[:, i]
        # print (time.time() - t)

        j = np.argmax(vals)
        x_k = np.roll(self.e_1, j)

        counts = np.zeros(x_k.shape).flatten()
        for i in range(self.T):
            j = np.argmax(grad_func(x_k))

            counts[j] += 1
            # st = time.time()
            # res = minimize_scalar(self.term, bounds=(0.0,1.0), method='bounded', args=(self.Q, x_k, j))
            # print('Optimization took {:.4f} secs'.format(time.time() - st))

            # val_1 =  np.sum(np.multiply(self.w - x_k), self.Q)
            # st = time.time()
            alpha = -np.dot(np.sum(np.einsum('ij,j->ij', self.Q, (x_k-np.roll(self.e_1, j)).flatten()), axis=1),
                           np.sum(np.einsum('ij,j->ij', self.Q, (self.w - x_k).flatten()), axis=1)) \
                    / np.linalg.norm(np.sum(np.einsum('ij,j->ij', self.Q, (x_k-np.roll(self.e_1, j)).flatten()),
                                            axis=1)) ** 2
            # print('Analytically took {:.4f}'.format(time.time() - st))
            x_k = x_k + alpha * (np.roll(self.e_1, j) - x_k)

        return self.P, x_k

    @staticmethod
    def main():
        n =20000; d=200
        P = np.random.randn(n, d)
        #P = np.vstack((P, 10000 * np.random.rand(2, 2)))
        w = np.ones((n, 1)) / n
        P = P - np.mean(P, 0)
        P = P / np.sqrt(np.sum(np.multiply(w, np.sum(P ** 2, axis=1))))
        frank_wolfe = FrankWolfeCoreset(P, w, 0.01)
        S, u = frank_wolfe.computeCoreset()
        print(np.linalg.norm(np.average(P, weights=w.flatten(), axis=0)- np.average(S, weights=u.flatten(), axis=0)))
        print('Our coreset is: {}'.format(S))


if __name__ == '__main__':
    ts = time.time()    
    FrankWolfeCoreset.main()
    te= time.time()
    print(te-ts)



