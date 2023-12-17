import numpy as np
import scipy

"""
author: Yvann Le Fay
"""


def minimize_matrix_input(f, init_matrix, method=None, options={'maxiter': 1}):
    shape = init_matrix.shape

    def _f(flattened_matrix):
        return f(flattened_matrix.reshape(shape))

    minimization = scipy.optimize.minimize(_f, init_matrix.flatten(), method=method, options=options)

    return minimization.x.reshape(shape), minimization.fun


def log_poisson_density(k, logparam):
    return - np.exp(logparam) + k * logparam - np.sum(np.log(np.arange(2, k + 1)))


def loss(adj, covariates, tau, alpha, beta):
    """
    Negative log-likelihood of the Poisson model.
    """

    """return -(np.einsum('ik,jl,kl,ij->', tau, tau, alpha, adj) + \
             np.einsum('ik,jl,ij,ij->', tau, tau, upper_covariates @ beta, adj) - \
             np.einsum('ik,jl,kl,ij->', tau, tau, np.exp(alpha), np.exp(upper_covariates @ beta)))
    """
    n = adj.shape[0]
    K = alpha.shape[0]
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(K):
                for l in range(K):
                    s += tau[i, k] * tau[j, l] * (
                            (alpha[k, l] + covariates[i, j].T @ beta) * adj[i, j] - np.exp(
                        alpha[k, l] + covariates[i, j].T @ beta))
    return s


def VE_step(adj, covariates, theta, tau):
    """
    VE step as described in the paper.
    """
    alpha, beta, nu = theta
    n = adj.shape[0]
    K = alpha.shape[0]
    log_params = np.array(
        [[[[alpha[k, l] + covariates[i, j].T @ beta for l in range(K)] for k in range(K)] for j in range(n)] for i in
         range(n)])
    log_poisson_terms = np.array([[
        [[log_poisson_density(adj[i, j], log_params[i, j, k, l]) for l in range(K)] for k in range(K)] for j in
        range(n)]
        for i in range(n)])
    diag_log_poisson_terms = np.array(
        [[[log_poisson_terms[i, i, k, l] for l in range(K)] for k in range(K)] for i in range(n)])
    new_log_tau = np.einsum('jl,ijkl->ik', tau, log_poisson_terms) - np.einsum('ikl->ik', diag_log_poisson_terms)
    new_log_tau -= np.max(new_log_tau, axis=1).reshape((-1, 1))
    new_tau = nu * np.exp(new_log_tau)
    new_tau /= np.sum(new_tau, axis=1).reshape((-1, 1))
    new_nu = np.mean(new_tau, axis=0)
    return new_nu, new_tau


def M_step(adj, upper_covariates, theta, tau):
    """
    M step as described in the paper.
    """
    alpha, beta, _ = theta
    f_alpha = lambda _alpha: -loss(adj, upper_covariates, tau, _alpha, beta)
    new_alpha, _ = minimize_matrix_input(f_alpha, alpha, method='Nelder-Mead')
    f_beta = lambda _beta: -loss(adj, upper_covariates, tau, new_alpha, _beta)
    new_beta, _ = minimize_matrix_input(f_beta, beta, method='Nelder-Mead')
    return new_alpha, new_beta


def VEM(adj, covariates, theta, tau, criterion=None):
    """
    Combining previous functions.
    """
    if criterion is None:
        criterion = lambda theta, init_theta, n_iter: n_iter < 2 and np.linalg.norm(theta[0] - init_theta[0]) > 1e-3 \
                                                      and np.linalg.norm(theta[1] - init_theta[1]) > 1e-3 \
                                                      and np.linalg.norm(theta[2] - init_theta[2]) > 1e-3
    n_iter = 0
    init_theta = theta
    init_tau = tau
    while criterion(theta, init_theta, n_iter) or n_iter == 0:
        init_theta = theta
        alpha, beta, _ = theta
        print(f'nll: {-loss(adj, covariates, tau, alpha, beta)}')
        n_iter += 1
        nu, tau = VE_step(adj, covariates, init_theta, init_tau)
        alpha, beta = M_step(adj, covariates, init_theta, init_tau)
        theta = (alpha, beta, nu)
        print(f'inferred theta: {theta}')
    print(f'terminal theta: {theta}')
    print(f'terminal nll: {-loss(adj, covariates, tau, alpha, beta)}')
    return theta
