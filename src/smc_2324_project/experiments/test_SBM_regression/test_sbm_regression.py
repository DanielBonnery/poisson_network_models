from src.smc_2324_project.variational.SBM_regression import VEM
from src.smc_2324_project.experiments.generate_from_model.generate_dataset import generate_network_params, \
    generate_covariates, \
    sample_from_network
import numpy as np


def main():
    # network
    alpha, beta, nu = generate_network_params()
    covariates = generate_covariates()
    # sample from network
    theta = (alpha, beta, nu)
    adj = sample_from_network(theta, covariates)
    n = adj.shape[0]
    K = alpha.shape[0]
    # init tau
    tau = np.random.normal(size=(n, K))
    tau /= np.sum(tau, axis=1).reshape((-1, 1))
    # VEM
    criterion = lambda alpha, beta, tau, nu, init_alpha, init_beta, init_tau, init_nu, n_iter: n_iter < 1000
    nu, tau, alpha, beta = VEM(adj, covariates, nu, tau, alpha, beta, criterion)


if __name__ == '__main__':
    main()
