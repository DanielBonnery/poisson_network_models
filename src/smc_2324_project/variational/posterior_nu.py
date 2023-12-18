import numpy as np


def sample_from_nu(e_0, tau, size=1):
    N = np.sum(tau, axis=0)
    nu = np.random.dirichlet(e_0 + N, size=size)
    return nu
