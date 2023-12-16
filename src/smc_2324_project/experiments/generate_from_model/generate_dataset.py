import numpy as np

# hyperparameters
n, k, p = 40, 2, 4
gamma_0 = np.array([1, 0, 3, 1.1, 2.2, 0.1, -0.3])
V_0 = np.eye(7)
e_0 = 3 * np.ones(2)


def generate_network_params():
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
    alpha, beta = gamma_to_alpha_beta(gamma)
    nu = np.random.dirichlet(e_0)

    return alpha, beta, nu


def gamma_to_alpha_beta(gamma):
    """
    Converts the flattened gamma to alpha and beta
    """
    alpha = np.zeros((k, k))
    alpha[np.triu_indices(k)] = gamma[:3]
    alpha = alpha + alpha.T - np.diag(np.diag(alpha))
    beta = gamma[3:]
    return alpha, beta


def generate_covariates():
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
    alpha, beta, nu = theta

    # sample Z
    Z = np.zeros((n, k))
    for i in range(n):
        Z_i = np.random.multinomial(1, nu)
        Z[i] = Z_i

    # sample Y
    Y = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for another_k in range(k):
                for l in range(k):
                    if (Z[i][another_k] == 1) and (Z[j][l] == 1):
                        lambda_ij = np.exp(X[i, j].dot(beta) + alpha[another_k, l])
                        Y[i, j] = np.random.poisson(lambda_ij)
    Y = Y + Y.T

    if return_Z:
        return Y, Z
    else:
        return Y
