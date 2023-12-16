import numpy as np
import scipy
from src.smc_2324_project.utils.utils import cast_to_upper

"""
author: Yvann Le Fay
"""


def log_poisson_density(k, logparam):
    return - np.exp(logparam) + k * logparam - np.sum(np.log(np.arange(2, k + 1)))


def loss(adj, upper_covariates, tau, alpha, beta):
    """
    Negative log-likelihood of the Poisson model.
    """

    return np.einsum('ik,jl,kl,ij->', tau, tau, alpha, adj) + \
           np.einsum('ik,jl,ij,ij->', tau, tau, upper_covariates @ beta, adj) - \
           np.einsum('ik,jl,kl,ij->', tau, tau, np.exp(alpha), np.exp(upper_covariates @ beta))


def VE_step(adj, covariates, nu, tau, alpha, beta):
    """
    VE step as described in the paper.
    """
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


def M_step(adj, upper_covariates, tau, alpha, beta):
    """
    M step as described in the paper.
    """
    f_alpha = lambda _alpha: loss(adj, upper_covariates, tau, _alpha, beta)
    alpha = scipy.optimize.newton(f_alpha, alpha)
    f_beta = lambda _beta: loss(adj, upper_covariates, tau, alpha, _beta)
    beta = scipy.optimize.newton(f_beta, beta)
    return alpha, beta


def VEM(adj, covariates, nu, tau, alpha, beta, criterion):
    """
    Combining previous functions.
    """
    if criterion is None:
        criterion = lambda alpha, beta, tau, nu, init_alpha, init_beta, init_tau, init_nu, n_iter: n_iter < 1000
    n_iter = 0
    upper_covariates = cast_to_upper(covariates)
    init_nu, init_tau, init_alpha, init_beta = nu, tau, alpha, beta
    while criterion(alpha, beta, tau, nu, init_alpha, init_beta, init_tau, init_nu, n_iter) or n_iter == 0:
        init_nu, init_tau, init_alpha, init_beta = nu, tau, alpha, beta
        n_iter += 1
        nu, tau = VE_step(adj, upper_covariates, init_nu, init_tau, init_alpha, init_beta)
        alpha, beta = M_step(adj, upper_covariates, init_tau, init_alpha, init_beta)
        print(f'iteration {n_iter}, nll {loss(adj, upper_covariates, tau, alpha, beta)}')
    return nu, tau, alpha, beta
