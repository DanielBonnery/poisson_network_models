import numpy as np
import scipy
from src.smc_2324_project.simulate.generate_dataset import gamma_to_alpha_beta

"""
author: Yvann Le Fay
"""


def minimize_matrix_input(f, init_matrix, method='Nelder-Mead', options={'maxiter': 10}):
    """
    Wrapping scipy.optimize.minimize to handle matrix inputs.
    """
    shape = init_matrix.shape

    def _f(flattened_matrix):
        return f(flattened_matrix.reshape(shape))

    minimization = scipy.optimize.minimize(_f, init_matrix.flatten(), method=method, options=options)

    return minimization.x.reshape(shape), minimization.fun


def log_poisson_density(k, logparam):
    return - np.exp(logparam) + k * logparam


def loss(adj, covariates, tau, gamma):
    """
    Negative log-likelihood of the Poisson model (up to a constant depending on tau, covariates and adj variables).
    """
    K = tau.shape[1]
    alpha, beta = gamma_to_alpha_beta(K, gamma)
    n = adj.shape[0]
    ind_i_lower_than_j = np.tri(n, n, -1).T
    s = np.einsum('ij,ik,jl,kl,ij->', ind_i_lower_than_j, tau, tau, alpha, adj)
    s += np.einsum('ij,ik,jl,ij,ij->', ind_i_lower_than_j, tau, tau, covariates @ beta, adj)
    s -= np.einsum('ij,ik,jl,kl,ij->', ind_i_lower_than_j, tau, tau, np.exp(alpha), np.exp(covariates @ beta))
    """s2 = 0
    for i in range(n):
        for j in range(n):
            for k in range(K):
                for l in range(K):
                    s2 += ind_i_lower_than_j[i, j] * (tau[i, k] * tau[j, l] * (
                            (alpha[k, l] + covariates[i, j].T @ beta) * adj[i, j] - np.exp(
                        alpha[k, l] + covariates[i, j].T @ beta)))
    assert s2 == s"""
    return -s


def VE_step(adj, covariates, gamma, nu, tau):
    """
    VE step as described in the paper.
    """
    n = adj.shape[0]
    K = tau.shape[1]
    alpha, beta = gamma_to_alpha_beta(K, gamma)
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


def M_step(adj, covariates, gamma, tau):
    """
    Minimize the negative log-likelihood with respect to gamma.
    """
    f_gamma = lambda _gamma: loss(adj, covariates, tau, _gamma)
    gamma, _ = minimize_matrix_input(f_gamma, gamma)
    return gamma


def VEM(adj, covariates, gamma, nu, tau, criterion=None):
    """
    Combining previous functions.
    """
    if criterion is None:
        criterion = lambda gamma, init_gamma, n_iter: n_iter < 10 and (np.linalg.norm(gamma - init_gamma) > 1e-2)
    n_iter = 0
    init_tau = tau
    init_gamma = gamma
    while criterion(gamma, init_gamma, n_iter) or n_iter == 0:
        init_gamma = gamma
        init_nu = nu
        n_iter += 1
        nu, tau = VE_step(adj, covariates, init_gamma, init_nu, init_tau)
        gamma = M_step(adj, covariates, init_gamma, init_tau)
    print(f'terminal gamma, nu: {gamma, nu}')
    return gamma, nu, tau
