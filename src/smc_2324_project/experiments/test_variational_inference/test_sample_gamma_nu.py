from src.smc_2324_project.simulate.generate_dataset import (
    generate_network_params,
    generate_covariates,
    sample_from_network,
    alpha_beta_to_gamma,
)
from src.smc_2324_project.variational.posterior_gamma import hessian, sample_from_gamma
from src.smc_2324_project.variational.posterior_nu import sample_from_nu
from src.smc_2324_project.variational.SBM_regression import VEM
import numpy as np


def main():
    np.random.seed(10)
    # hyperparameters
    n, K, p = 40, 3, 4
    gamma_0 = np.array([1, 0, 3, 1.1, 1.5, 0.2, -0.1, 2.2, 0.1, -0.3])
    assert gamma_0.shape[0] == (K * (K + 1)) // 2 + p
    V_0 = np.eye(10) * 0.1
    assert V_0.shape[0] == p + (K * (K + 1)) // 2
    e_0 = 3 * np.ones(3)
    assert e_0.shape[0] == K
    # network
    theta = generate_network_params(K, gamma_0, V_0, e_0)
    covariates = generate_covariates(n, p)
    # sample from network
    adj = sample_from_network(theta, covariates)
    gamma = alpha_beta_to_gamma(theta[0], theta[1])
    print(f"true_gamma, nu: {gamma, theta[2]}")
    # random init. Does it make sense ?
    # init tau
    tau = np.random.uniform(0, 1, size=(n, K))
    tau /= np.sum(tau, axis=1).reshape((-1, 1))
    # init nu
    nu = np.mean(tau, axis=0)
    # VEM
    _fake_alpha = np.random.normal(size=int(K * (K + 1) / 2))
    fake_alpha = np.zeros((K, K))
    fake_alpha[np.triu_indices(K)] = _fake_alpha
    fake_alpha += fake_alpha.T - np.diag(np.diag(fake_alpha))
    fake_beta = np.random.normal(size=theta[1].shape)
    fake_gamma = alpha_beta_to_gamma(fake_alpha, fake_beta)
    print(f"fake_gamma, nu: {fake_gamma, nu}")
    inferred_gamma, inferred_nu, inferred_tau = VEM(
        adj, covariates, fake_gamma, nu, tau
    )
    hess = hessian(adj, covariates, tau, fake_gamma)
    samples_gamma = sample_from_gamma(hess, gamma_0, V_0, inferred_gamma, 10)
    samples_nu = sample_from_nu(e_0, inferred_tau, 10)
    print(samples_gamma, samples_nu)


if __name__ == "__main__":
    main()
