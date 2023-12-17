import numpy as np
import scipy
import sklearn.linear_model

from src.smc_2324_project.utils.utils import cast_to_upper

"""
author: Yvann Le Fay
"""


def minimize_matrix_input(f, init_matrix, method=None, options={'maxiter': 10}):
    shape = init_matrix.shape

    def _f(flattened_matrix):
        return f(flattened_matrix.reshape(shape))

    minimization = scipy.optimize.minimize(_f, init_matrix.flatten(), method=method, options=options)

    return minimization.x.reshape(shape), minimization.fun


def log_poisson_density(k, logparam):
    return - np.exp(logparam) + k * logparam


def loss(adj, upper_covariates, tau, alpha, beta):
    """
    Negative log-likelihood of the Poisson model.
    """

    return -(np.einsum('ik,jl,kl,ij->', tau, tau, alpha, adj) + \
             np.einsum('ik,jl,ij,ij->', tau, tau, upper_covariates @ beta, adj) - \
             np.einsum('ik,jl,kl,ij->', tau, tau, np.exp(alpha), np.exp(upper_covariates @ beta)))
    """n = adj.shape[0]
    K = alpha.shape[0]
    return -np.sum(np.array([[[[tau[i, k] * tau[j, l] * (
            (alpha[k, l] + upper_covariates[i, j].T @ beta) * adj[i, j] - np.exp(
        alpha[k, l] + upper_covariates[i, j].T @ beta)) for l in range(K)] for k in range(K)] for j in range(n)] for
                             i in range(n)]))"""


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
    # new_log_tau -= np.max(new_log_tau, axis=1).reshape((-1, 1))
    new_tau = nu * np.exp(new_log_tau)
    new_tau /= np.sum(new_tau, axis=1).reshape((-1, 1))
    new_nu = np.mean(new_tau, axis=0)
    return new_nu, new_tau


def M_step(adj, upper_covariates, theta, tau):
    """
    M step as described in the paper.
    """
    alpha, beta, _ = theta
    f_alpha = lambda _alpha: loss(adj, upper_covariates, tau, _alpha, beta)
    new_alpha, _ = minimize_matrix_input(f_alpha, alpha)
    f_beta = lambda _beta: loss(adj, upper_covariates, tau, new_alpha, _beta)
    new_beta, _ = minimize_matrix_input(f_beta, beta)
    return new_alpha, new_beta


def VEM(adj, covariates, theta, tau, criterion=None):
    """
    Combining previous functions.
    """
    if criterion is None:
        criterion = lambda theta, init_theta, n_iter: n_iter < 10 and np.linalg.norm(theta[0] - init_theta[0]) > 1e-3 \
                                                      and np.linalg.norm(theta[1] - init_theta[1]) > 1e-3 \
                                                      and np.linalg.norm(theta[2] - init_theta[2]) > 1e-3
    n_iter = 0
    upper_covariates = cast_to_upper(covariates)
    init_theta = theta
    init_tau = tau
    while criterion(theta, init_theta, n_iter) or n_iter == 0:
        init_theta = theta
        n_iter += 1
        nu, tau = VE_step(adj, upper_covariates, init_theta, init_tau)
        alpha, beta = M_step(adj, upper_covariates, init_theta, init_tau)
        theta = (alpha, beta, nu)
        print(f'inferred theta: {theta}')
        print(f'nll: {loss(adj, upper_covariates, tau, alpha, beta)}')
        # print(f'iteration {n_iter}, nll {loss(adj, upper_covariates, tau, alpha, beta)}')
    print(f'terminal theta: {theta}')
    print(f'terminal nll: {loss(adj, upper_covariates, tau, alpha, beta)}')
    return theta
