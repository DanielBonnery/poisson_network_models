import numpy as np


def from_flat_indice_to_triu_indices(i, size):
    return i // size, i // size + (i % size)


def hessian(adj, covariates, tau, theta):
    """
    Compute the Hessian H_J^{\gamma} used in Eq. 9.
    """
    alpha, beta, _ = theta
    n = adj.shape[0]
    K = alpha.shape[0]
    n_K = int(K * (K + 1) / 2)
    d = covariates.shape[-1]  # or p according to Paul's notations
    hess = np.zeros((n_K + d, n_K + d))
    log_params = np.array(
        [[[[alpha[k, l] + covariates[i, j].T @ beta for l in range(K)] for k in range(K)] for j in range(n)] for i in
         range(n)])
    params = np.exp(log_params)
    hess[:n_K, :n_K] = np.diag(
        [-np.sum([[params[i, j, *from_flat_indice_to_triu_indices(index, K)] for j in range(n)] for i in range(n)])
         for index
         in range(n_K)])
    for index1 in range(d):
        for index2 in range(d):
            hess[n_K + index1, n_K + index2] = - np.sum(
                [[[[tau[i, k] * tau[j, l] * covariates[i, j, index1] * covariates[i, j, index2] * params[i, j, k, l] for
                    l in range(K)] for k in range(K)] for j in range(n)] for i in range(n)])
        # could be optimized
    # cross terms
    for index1 in range(n_K):
        for index2 in range(d):
            i1, i2 = from_flat_indice_to_triu_indices(index1, K)
            cross_term = - np.sum(
                [[tau[i, i1] * tau[j, i2] * covariates[i, j, index2] * params[i, j, i1, i2] for j in range(n)] for i in
                 range(n)])
            hess[index1, n_K + index2] = cross_term
            hess[n_K + index2, index1] = cross_term
    return hess


def sample_from_gamma(hess, gamma0, V0, inferred_gamma):
    """
    Samples from the Gaussian posterior of gamma.
    """
    invV0 = np.linalg.inv(V0)
    cov = np.linalg.inv((invV0 - hess))
    mean = cov @ (invV0 @ gamma0 - hess @ inferred_gamma)
    return np.random.multivariate_normal(mean, cov)
