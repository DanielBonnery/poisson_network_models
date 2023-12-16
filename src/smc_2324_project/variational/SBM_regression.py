import numpy as np
import scipy
from src.smc_2324_project.utils.utils import cast_to_upper


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
    new_tau = nu * np.exp(new_log_tau)
    new_tau /= np.sum(new_tau, axis=1).reshape((-1, 1))
    new_nu = np.mean(new_tau, axis=0)
    return new_tau, new_nu


def M_step(adj, upper_covariates, tau, alpha, beta):
    f_alpha = lambda _alpha: loss(adj, upper_covariates, tau, _alpha, beta)
    res_alpha = scipy.optimize.newton(f_alpha, alpha)
    alpha = res_alpha.x
    f_beta = lambda _beta: loss(adj, upper_covariates, tau, _beta, beta)
    res_beta = scipy.optimize.newton(f_beta, beta)
    beta = res_beta.x
    return alpha, beta


def VEM_step(adj, covariates, nu, tau, alpha, beta, criterion):
    n_iter = 0
    upper_covariates = cast_to_upper(covariates)
    while criterion(alpha, beta, n_iter) or n_iter == 0:
        init_nu, init_tau, init_alpha, init_beta = tau, nu, alpha, beta
        n_iter += 1
        tau, nu = VE_step(adj, upper_covariates, init_nu, init_tau, init_alpha, init_beta)
        alpha, beta = M_step(adj, upper_covariates, init_tau, init_alpha, init_beta)
    return alpha, beta, nu, tau
