from src.smc_2324_project.variational.SBM_regression import VEM
from src.smc_2324_project.experiments.generate_from_model.generate_dataset import generate_network_params, \
    generate_covariates, \
    sample_from_network
import numpy as np


def main():
    # hyperparameters
    n, k, p = 40, 2, 4
    gamma_0 = np.array([1, 0, 3, 1.1, 2.2, 0.1, -0.3])
    V_0 = np.eye(7) * 0.1
    e_0 = 3 * np.ones(2)
    # network
    theta = generate_network_params(k, gamma_0, V_0, e_0)
    covariates = generate_covariates(n, p)
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
