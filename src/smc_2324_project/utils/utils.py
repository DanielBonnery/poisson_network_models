import numpy as np


def cast_to_upper(covariates):
    """
    Cast covariates to upper triangular covariates,
    """
    dim = covariates.shape[-1]
    tri = np.tri(covariates.shape[0], covariates.shape[1], -1).T
    return np.moveaxis(np.array([covariates[..., j] * tri for j in range(dim)]), 0, -1)
