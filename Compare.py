import numpy as np
import FastEpsCoreset as FEC
import Utils
import time
import copy
from scipy.spatial import distance_matrix
from AlaaBoost import updated_cara
# import FOR17
from FastEpsCoreset import sparseEpsCoreset
from attainCoresetByDan import attainCoresetByDanV1
from graphPlotter import GraphPlotter


class Compare(object):

    computeOptValue = (lambda P, w: np.sum(np.multiply(w, np.sum((P - np.average(P, axis=0, weights=w)) ** 2, 1))))
    computeOnAllData = (lambda P,w,q: np.sum(np.multiply(w, np.sum((P - q) ** 2, 1))))
    computeOptValueSVD = (lambda P, w,k: np.sqrt(np.sum(np.linalg.svd(np.multiply(w[:, np.newaxis], P), full_matrices=False)[1][:k]**2)))

    def __init__(self, file_name, compare_SVD=False, k=2):
        self.P = Utils.readDataset(file_name)
        self.k = k
        self.compare_SVD = compare_SVD
        print('n = {}, d = {}'.format(self.P.shape[0], self.P.shape[1]))
        self.weights = np.ones((self.P.shape[0], ))
        if not self.compare_SVD:
            self.Q, self.W, self.mu, self.sigma = Utils.getNormalizedWeightedSet(self.P, self.weights)
        else:
            self.Q = Utils.preprocessDataForSVDComaprison(self.P, self.k)
            self.W = np.ones((self.Q.shape[0], ))
            self.sigma = 1
        self.legend = ['Uniform Sampling', 'Sensitivity Sampling', 'Caratheodory', 'Dan ICML2016',
                       'Our slow coreset', 'Our fast Coreset']

        self.file_name = file_name.split('.')[0]
        Utils.createDirectory(self.file_name)
        self.sampling_algorithms = [
            lambda sample_size, sensitivity: self.computeCoreset(self.Q, sensitivity, sample_size),
            lambda sample_size: updated_cara(self.Q, self.W, None),
            lambda sample_size: attainCoresetByDanV1(self.Q, self.W, 1.0/sample_size),
            lambda sample_size: sparseEpsCoreset(self.Q, self.W, 1.0/sample_size, faster=False),
            lambda sample_size: sparseEpsCoreset(self.Q, self.W, 1.0 / sample_size, faster=True)
        ]

        if self.compare_SVD:
            self.legend = self.legend[4:]
            self.sampling_algorithms = self.sampling_algorithms[3:]
            self.opt_value = Compare.computeOptValueSVD(self.P, self.W, self.k)
        else:
            self.opt_value = Compare.computeOptValue(self.Q, self.W)

        self.graph_plotter = GraphPlotter()

    def tightBoundSensitivity(self):
        return np.multiply(self.W / np.sum(self.W), (1 + np.sum(self.Q ** 2, 1) / self.sigma))

    def computeCoreset(self, P, sensitivity, sampleSize):
        """
        :param P: A matrix of nxd points where the last column is the labels.
        :param sensitivity: A vector of n entries (number of points of P) which describes the sensitivity of each point.
        :param sampleSize: An integer describing the size of the coreset.
        :param weights: A weight vector of the data points (Default value: None)
        :return: A subset of P (the datapoints alongside their respected labels), weights for that subset and the
        time needed for generating the coreset.
        """

        start_time = time.time()

        # Compute the sum of sensitivities.
        t = np.sum(sensitivity)

        # The probability of a point prob(p_i) = s(p_i) / t
        probability = sensitivity.flatten() / t

        # The number of points is equivalent to the number of rows in P.
        n = P.shape[0]

        # initialize new seed
        np.random.seed()

        # Multinomial distribution.
        indxs = np.random.choice(n, sampleSize, p=probability.flatten())

        # Compute the frequencies of each sampled item.
        hist = np.histogram(indxs, bins=range(n))[0].flatten()
        indxs = copy.deepcopy(np.nonzero(hist)[0])

        # Select the indices.
        S = self.P[indxs, :]

        # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
        weights = np.asarray(np.multiply(self.W[indxs], hist[indxs]), dtype=float).flatten()

        # Compute the weights of the coreset
        weights = np.multiply(weights, 1.0 / (probability[indxs] * sampleSize))
        time_taken = time.time() - start_time

        # return S[:, :-1], S[:, -1], weights, timeTaken
        return S, weights, time_taken

    def computeError(self, weighted_set):
        start_time = time.time()
        if self.compare_SVD:
            W = np.zeros((self.P.shape[0], ))
            W[weighted_set[-1]] = weighted_set[1]
            _, _, V = np.linalg.svd(np.multiply(W[:, np.newaxis], self.P), full_matrices=False)
            return np.linalg.norm(np.dot(self.P, V[:self.k, :].T),ord='fro') / self.opt_value - 1
        else:
            mean_on_coreset = np.average(weighted_set[0], axis=0, weights=weighted_set[1].flatten())

        return Compare.computeOnAllData(self.Q, self.W, mean_on_coreset) / self.opt_value - 1,\
               time.time() - start_time + weighted_set[2]

    def applySamplingAndAttainError(self, alg, sample_size, sensitivity=None):
        if alg < 2 and not self.compare_SVD:
            results = [self.computeError(self.sampling_algorithms[0](sample_size, sensitivity))
                       for i in range(Utils.REPS)]
            return np.mean([results[i][0] for i in range(len(results))]), \
                                    np.mean([results[i][1] for i in range(len(results))])
        else:
            return self.computeError(self.sampling_algorithms[alg - 1 if not self.compare_SVD else alg](sample_size))

    def applyComaprison(self):

        mean_error = np.empty((len(self.legend), Utils.NUM_SAMPLES))
        mean_time = np.empty((len(self.legend), Utils.NUM_SAMPLES))
        samples = Utils.generateSampleSizes(self.P.shape[0], self.compare_SVD)

        if not self.compare_SVD:
            sensitivity = self.tightBoundSensitivity()
            all_sensitivity = np.vstack((np.ones(sensitivity.shape), sensitivity))
            mean_error_cara, mean_time_cara = self.applySamplingAndAttainError(2, 0, None)

        
        for idx, sample_size in enumerate(samples):
            print('Sample size: {}'.format(sample_size))
            for alg in range(len(self.legend)):
                if alg != 2 or self.compare_SVD:
                    mean_error[alg, idx], mean_time[alg, idx] = \
                        self.applySamplingAndAttainError(alg, sample_size if not self.compare_SVD
                        else 25 * self.k**2 * sample_size ** 2,  all_sensitivity[alg, :]
                        if alg < 2 and not self.compare_SVD else None)
                else:
                    mean_error[alg, idx], mean_time[alg,idx] = mean_error_cara, mean_time_cara

        file_path = r'results/{}/Results_{}.npz'.format(self.file_name, self.file_name)
        if self.compare_SVD:
            file_path = file_path.replace('.npz', '_SVD.npz')
        np.savez(file_path, mean_error=mean_error, mean_time=mean_time)

        file_path = r'results/{}/{}-{}.pdf'.format('Synthetic', 'Synthetic', 'error')
        self.graph_plotter.plotGraph(samples, mean_error, self.legend,
                                     'Synthetic', 'sample size', r'$\varepsilon$', file_path)


        file_path = r'results/{}/{}-{}.pdf'.format('Synthetic', 'Synthetic', 'time')
        self.graph_plotter.plotGraph(samples, mean_time, self.legend, 'Synthetic', 'sample size', 'Overall time (secs)', file_path)



        print('******************** ERROR *********************')
        print('first row: Uniform\n second row: sensitivity\n third row: Cara\n forth row: Us')
        print(mean_error)
        print('******************** TIME ************************')
        print(mean_time)


    @staticmethod
    def main():
        n = 100000
        d = 5
        k = 2
        A = np.random.randn(n, d)
        compare_SVD = False

        np.save(r'datasets/Synthetic.npy', A)
        file_name = 'Synthetic.npy'
        main_runner = Compare(file_name, compare_SVD, k)
        main_runner.applyComaprison()

if __name__ == '__main__':
    Compare.main()



