import numpy as np
import scipy as sp
from scipy.linalg import null_space
import FrankWolfeCoreset as FWC
import copy

def checkIfPointOnSegmen(p, start_line, end_line):
	v = (p - end_line) / np.linalg.norm(p - end_line)
	z = (start_line - end_line) / np.linalg.norm(start_line - end_line)

	if np.linalg.norm(v-z) <= 1e-10 or np.linalg.norm(v+z) <= 1e-10:
		return True
	
	return False


def attainCoresetByDan(P, u, eps):
	E_u = np.sum(np.multiply(P, u), axis=0)
	x = np.sum(np.multiply(u.flatten(), np.sqrt(np.sum((P - E_u) ** 2,axis=1))))
	lifted_P = np.hstack((P - E_u, x * np.ones((P.shape[0], 1))))
	v = np.sum(np.multiply(u.flatten(), np.sqrt(np.sum(Q**2, axis=1))))
	Q = np.multiply(P, 1/np.linalg.norm(lifted_P, ord=2, axis=1)[:, np.newaxis])
	s = np.multiply(u.flatten(), 1/v * np.linalg.norm(lifted_P, ord=2, axis=1))

	last_entry_vec = np.zeros(1, lifted_P.shape[1])
	last_entry_vec[-1] = x / v

	H = Q - last_entry_vec

	tau = v / int(np.sqrt(1/eps))
	alpha = 2 * (1 + 2 * (1 + tau ** 2) / (1 - tau) ** 2)

	beta = int(np.ceil(alpha / eps))
	h = np.empty((beta, H.shape[1]))
	c_i = copy.deepcopy(h)
	c_i[0,:] = np.random.choice(np.arange(P.shape[0]))
	origin = np.zeros((H.shape[1], ))
	for i in range(beta-1):
		h[i, :] = H[np.argmax(np.linalg.norm(H - c_i[i, :], ord=2, axis=1)), :]
		orth_line_segment = null_space(h[i, :] - c_i[i, :])
		project_origin = np.dot(origin - c_i[i, :], orth_line_segment)
		if checkIfPointOnSegmen(project_origin, c_i[i, :], h[i, :]):
			c_i[i+1, :] = project_origin
		else:
			dist1, dist2 = np.linalg.norm(project_origin - c_i[i, :]), np.linalg.norm(project_origin - h[i, :])
			c_i[i+1, :] = h[i, :] if dist2 < dist1 else c_i[i, :]

	_, w_prime = FWC.FrankWolfeCoreset(Q, s[:, np.newaxis], eps).computeCoreset()

	w_double_prime = np.multiply(v * w_prime.flatten(), 1/np.linalg.norm(lifted_P, ord=2, axis=1))
	w = w_double_prime / np.sum(w_double_prime)

	S = P[np.where(w>0)[0], :]

	return S, w[np.where(w>0)[0]]


if __name__ == '__main__':
	n = 1000
	d = 10
	P = np.random.randn(n, d)
	w = np.ones((n, 1)) / n
	P = P - np.mean(P, 0)
	P = P / np.sqrt(np.sum(np.multiply(w, np.sum(P ** 2, axis=1))))
	S, u = attainCoresetByDan(P, w, 0.01)
	print(np.linalg.norm(np.average(P, weights=w.flatten(), axis=0) - np.average(S, weights=u.flatten(), axis=0)))
	print('Our coreset is: {}'.format(S))


