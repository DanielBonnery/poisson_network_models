import numpy as np

n, k, p = 40, 2, 4
gamma_0 = np.array([1, 0, 3, 1.1, 2.2, 0.1, -0.3])
V_0 = np.eye(7)
e_0 = 3 * np.ones(2)


def generate_network_params(k=k, gamma_0=gamma_0, V_0=V_0, e_0=e_0):
    """
    Generates the parameters alpha, beta and nu of a network model.

    Fixed hyperparameters:
        n = 40: number of nodes
        k = 2: number of communities
        p = 4: number of covariates

    Parameters shapes:
        alpha: (k, k) natively but symmetric matrix, so flattened to k*(k+1)/2
        beta: (p, 1)
        nu: (k, 1)
        gamma: flattened version of (alpha,beta), so k*(k+1)/2 + p

    Samples from the following prior:
        alpha, beta ~ N(gamma_0, V_0)
        nu ~ Dirichlet(e_0)"""

    # sample
    gamma = np.random.multivariate_normal(gamma_0, V_0)
    alpha, beta = gamma_to_alpha_beta(k, gamma)
    nu = np.random.dirichlet(e_0)

    return alpha, beta, nu


def gamma_to_alpha_beta(k, gamma):
    """
    Converts the flattened gamma to alpha and beta
    """
    alpha = np.zeros((k, k))
    alpha[np.triu_indices(k)] = gamma[: k * (k + 1) // 2]
    alpha = alpha + alpha.T - np.diag(np.diag(alpha))
    beta = gamma[k * (k + 1) // 2 :]
    return alpha, beta


def generate_covariates(n=n, p=p):
    """
    Generates the covariates X.
    Fixed hyperparameters:
        n = 40: number of nodes
        p = 4: number of covariates
    X has shape (n, n, p)
    """
    X = np.random.normal(0, 1, (n, n, p))
    return X


def sample_from_network(theta, X, return_Z=False):
    """
    Samples from the network model with parameters theta and covariates X.
    Returns the counts matrix Y (shape (n,n)).
    theta = (alpha, beta, nu), alpha and beta being in their matrix form.
    """
    # parameters
    alpha, beta, nu = theta
    n = X.shape[0]
    k = alpha.shape[0]

    # sample Z
    Z = np.zeros((n, k))
    Z_idx = np.random.choice(k, size=n, p=nu)
    Z[range(n), Z_idx] = 1.0
    assert np.all(Z.sum(axis=1) == 1)

    # sample Y
    Y = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for another_k in range(k):
                for l in range(k):
                    if (Z[i][another_k] == 1) and (Z[j][l] == 1):
                        lambda_ij = np.exp(X[i, j].dot(beta) + alpha[another_k, l])
                        Y[i, j] = np.random.poisson(lambda_ij)
                        break
                break

    Y = Y + Y.T

    assert not np.any(np.isnan(Y))
    if return_Z:
        return Z, Y
    else:
        return Y
