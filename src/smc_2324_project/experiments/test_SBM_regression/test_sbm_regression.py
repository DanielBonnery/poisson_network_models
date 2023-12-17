from src.smc_2324_project.variational.SBM_regression import VEM
from src.smc_2324_project.experiments.generate_from_model.generate_dataset import generate_network_params, \
    generate_covariates, \
    sample_from_network
import numpy as np


def main():
    # network
    theta = generate_network_params()
    covariates = generate_covariates()
    # sample from network
    adj = sample_from_network(theta, covariates)
    n = adj.shape[0]
    K = theta[0].shape[0]
    # init tau
    tau = np.random.normal(size=(n, K))
    tau /= np.sum(tau, axis=1).reshape((-1, 1))
    # init nu
    nu = np.random.normal(size=theta[2].shape)
    nu /= np.sum(nu)
    # VEM
    fake_theta = (
        np.random.normal(size=theta[0].shape), np.random.normal(size=theta[1].shape),
        nu)
    print(f'true_theta: {theta}')
    print(f'fake_theta: {fake_theta}')
    inferred_theta = VEM(adj, covariates, theta, tau)
    return inferred_theta


if __name__ == '__main__':
    main()
