import numpy as np
from scipy import optimize
import copy
from scipy.stats import poisson, multivariate_normal
from particles import resampling as rs
from particles.smc_samplers import AdaptiveTempering

from src.smc_2324_project.simulate.generate_dataset import *

### Distributions (target and so on) ###


def convert_categorical_to_one_hot(Z_categorical, k):
    """Convert a categorical array to a one-hot array.
    Z has shape (N, n) and its values are in {0, ..., k-1}.
    Z_one_hot has shape (N, n, k) and its values are in {0, 1}."""
    N, n = Z_categorical.shape
    Z = np.zeros((N, n, k))
    i, j = np.meshgrid(np.arange(N), np.arange(n), indexing="ij")
    Z[i, j, Z_categorical.astype(int)] = 1
    return Z


def compute_llh_marginal(particles, X, Y, k):
    """
    Compute the marginal loglikelyhoods given the data for a list of particles.
    Vectorized.
    """
    if particles.ndim == 0:
        return compute_llh_marginal(np.ndarray([particles]), X, Y, k)
    # extract
    Z_categorical, theta = particles["Z"], particles["theta"]
    N, n = Z_categorical.shape
    Z = convert_categorical_to_one_hot(Z_categorical, k)
    gamma, nu = theta["gamma"], theta["nu"]
    alpha, beta = gamma_to_alpha_beta(k, gamma)

    # compute logpz
    logpz = np.sum(np.log(np.einsum("mnk,mk->mn", Z, nu)), axis=1)
    assert not np.any(np.isinf(logpz)) and not np.any(np.isnan(logpz))
    assert logpz.shape == (N,)

    # compute lambda_poisson
    interaction_terms = np.einsum("mnk,mkl->mnl", Z, alpha)
    interaction_terms = np.einsum("mnl,mul->mnu", interaction_terms, Z)
    covariate_terms = np.einsum("nvp,mp->mnv", X, beta)
    lambda_poisson = np.exp(interaction_terms + covariate_terms)
    assert lambda_poisson.shape == (N, n, n)
    indices = np.triu_indices(n, k=1)

    # compute logpy
    Y_flattened = np.tile(Y[indices[0], indices[1]], (N, 1)).flatten()
    lambda_poisson_flattened = lambda_poisson[:, indices[0], indices[1]].flatten()
    logpy = poisson.logpmf(Y_flattened, lambda_poisson_flattened)
    assert logpy.shape == (N * n * (n - 1) // 2,)
    logpy = logpy.reshape((N, n * (n - 1) // 2)).sum(axis=1)
    assert np.all(logpz + logpy < 0)
    return logpz + logpy


def define_llh_target(prior, X, Y, k):
    """Returns the loglikelyhood function corresponding to the posterior."""

    def compute_llh_target(particles):
        """Y and X to be defined globally ?"""
        return compute_llh_marginal(particles, X, Y, k) + prior.logpdf(particles)

    return compute_llh_target


### Kernel ###


class FixedLenMCMCSequence:
    """Base class for a (fixed length or adaptive) sequence of MCMC steps."""

    def __init__(self, mcmc=None, len_chain=10):
        self.mcmc = ArrayRandomWalk() if mcmc is None else mcmc
        self.nsteps = len_chain - 1

    def calibrate(self, W, x):
        self.mcmc.calibrate(W, x)

    def __call__(self, x, target):
        for _ in range(self.nsteps):
            x = self.mcmc(x, target)
        return x


def dirichlet_sample(alphas):
    """
    Generate samples from an array of alpha distributions.
    """
    r = np.random.standard_gamma(alphas)
    return r / r.sum(-1, keepdims=True)


def sample_nu(x, e_0, e_tilde, rho, k):
    """Sample nu.
    All particles at once.
    Inplace."""
    Z_categorical = x.theta["Z"]
    assert Z_categorical.ndim == 2
    N, n = Z_categorical.shape
    Z = convert_categorical_to_one_hot(Z_categorical, k)
    e = rho * (e_0 + np.sum(Z, axis=1)) + (1 - rho) * e_tilde
    nu = dirichlet_sample(e)
    x.theta["theta"]["nu"] = nu


def compute_gamma_var(x):
    """Compute the variance of gamma.
    All particles at once."""
    gamma = x.theta["theta"]["gamma"]
    empirical_gamma_var = gamma.T @ gamma / len(gamma)
    return empirical_gamma_var


def sample_gamma(x, gamma_var, target):
    """Sample gamma.
    In place."""
    x_proposal = copy.deepcopy(
        x
    )  # TODO: check if deepcopy is necessary or can be mutualized instance-wise
    gamma = x.theta["theta"]["gamma"]
    N, len_gamma = gamma.shape

    # proposal step
    gamma_step = np.random.multivariate_normal(np.zeros(len_gamma), gamma_var, size=N)
    x_proposal.theta["theta"]["gamma"] = gamma + gamma_step

    # compute logratio
    target(x)
    target(x_proposal)
    logratio = x_proposal.lpost - x.lpost
    ratio = np.exp(logratio)
    u = np.random.rand(N)

    # accept or reject
    x.theta["theta"]["gamma"][u < ratio] = x_proposal.theta["theta"]["gamma"][u < ratio]

    # compute acceptance rate
    acc_rate = np.mean(u < ratio)
    return acc_rate


def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def sample_Z(x, target, k):
    """Sample Z.
    In place."""
    x_proposal = copy.deepcopy(x)
    Z_categorical = x.theta["Z"]
    N, n = Z_categorical.shape

    # Z by Z, k by k lk computation then sample, parrallelized over particles
    for Z_idx in range(n):  # iterating over Z rows
        # beginning a new sweep
        Z_i_proposal_lh = np.zeros((N, k))

        # iterating over Z values
        for Z_val in range(k):
            x_proposal.theta["Z"][:, Z_idx] = Z_val

            # compute logprobas
            target(x_proposal)
            Z_i_proposal_lh[:, Z_val] = x_proposal.lpost

        # sample Z
        Z_i_proposal_lh = np.exp(Z_i_proposal_lh - np.max(Z_i_proposal_lh))
        Z_i_proposal_lh /= np.sum(Z_i_proposal_lh, axis=1, keepdims=True)

        # sample Z
        x.theta["Z"][:, Z_idx] = random_choice_prob_index(Z_i_proposal_lh, axis=1)
        x_proposal.theta["Z"] = x.theta["Z"]


def update_acc_rate(x, acc_rate):
    """Update the acc_rate attribute of x.
    In place."""
    if "acc_rates" not in x.shared:
        x.shared["acc_rates"] = [acc_rate]
    else:
        x.shared["acc_rates"].append(acc_rate)


class CustomGibbs:
    """A custom Gibbs step of the kernel.
    Beware, in that case, x is a ThetaParticles object."""

    def __init__(self, k, e_0, e_tilde):
        self.k = k
        self.e_0 = e_0
        self.e_tilde = e_tilde

    def calibrate(self, W, x):
        self.rho = x.shared["exponents"][-1]
        self.gamma_var = compute_gamma_var(x)

    def __call__(self, x, target):
        # target update the lpost attribute of x
        sample_nu(x, self.e_0, self.e_tilde, self.rho, self.k)
        len_gamma = x.theta["theta"]["gamma"].shape[1]
        mean_acc_rate_gamma = sample_gamma(
            x, 2.38**2 / len_gamma * self.gamma_var, target
        )
        sample_Z(x, target, self.k)
        update_acc_rate(x, mean_acc_rate_gamma)
        update_acc_rate(x, mean_acc_rate_gamma)
        return x


### Flexibe tau 2 ###


class FlexibeAdaptiveTempering(AdaptiveTempering):
    """Adaptive tempering with a flexible tempering_step_size."""

    def __init__(self, *args, **kwargs):
        self.tempering_step_size = kwargs["tempering_step_size"]
        del kwargs["tempering_step_size"]
        super().__init__(*args, **kwargs)

    def logG(self, t, xp, x):
        ESSmin = (1 - self.tempering_step_size) * x.N
        f = lambda e: rs.essl(e * x.llik) - ESSmin
        epn = x.shared["exponents"][-1]
        if f(1.0 - epn) > 0:  # we're done (last iteration)
            delta = 1.0 - epn
            new_epn = 1.0
            # set 1. manually so that we can safely test == 1.
        else:
            delta = optimize.brentq(f, 1.0e-12, 1.0 - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        x.shared["exponents"].append(new_epn)
        return self.logG_tempering(x, delta)
