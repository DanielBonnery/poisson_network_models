from collections import OrderedDict
import numpy as np
from scipy.stats import dirichlet
from particles.distributions import *

### Distributions ###


class CategoricalFix(Categorical):
    """Categorical distribution.
    Fix int casting during logpdf computation.
    """

    dtype = int

    def __init__(self, p=None):
        super().__init__(p=p)

    def logpdf(self, x):
        lp = np.log(self.p)
        d = lp.shape[-1]
        choices = [lp[..., k] for k in range(d)]
        x = np.array(x).astype(int)  # to be solved
        return np.choose(x, choices)


class Dirichlet(ProbDist):
    """Dirichlet distribution.

    Parameters
    ----------
    alpha: array-like
        Parameters of the Dirichlet distribution

    Note:
        Returns the Dirichlet distribution.

    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = np.array(alpha)
        self.dim = len(alpha)

    def rvs(self, size=1):
        return dirichlet.rvs(self.alpha, size=size)

    def logpdf(self, x):
        if x.ndim == 2:
            return dirichlet.logpdf(x.T, self.alpha)
        return dirichlet.logpdf(x, self.alpha)

    def ppf(self, u):
        """Use the ppf of k gammas and return the normalized vector."""
        raise NotImplementedError


# class Multinomial(ProbDist):
#     """Multinomial distribution.

#     Parameters
#     ----------
#     p: array-like
#         Parameters of the Multinomial distribution
#     """

#     def __init__(self, p):
#         super().__init__()
#         self.p = np.array(p)
#         self.dim = len(p)

#     def rvs(self):
#         x = np.zeros_like(self.p)
#         x[np.random.choice(len(self.p), p=self.p)] = 1
#         return x

#     def logpdf(self, x):
#         idx = np.where(x != 0)[0]
#         return np.log(self.p[idx])

#     def ppf(self, u):
#         raise NotImplementedError

### Prior and base_dist ###


def define_theta_prior(gamma_0, V_0, e_0):
    theta_prior_dict = OrderedDict()
    theta_prior_dict["gamma"] = MvNormal(loc=gamma_0, cov=V_0)
    theta_prior_dict["nu"] = Dirichlet(e_0)
    theta_prior = StructDist(theta_prior_dict)
    return theta_prior


def define_prior(theta_prior: StructDist, n: int):
    prior_dict = OrderedDict()
    prior_dict["theta"] = theta_prior
    prior_dict["Z"] = Cond(
        lambda x: IndepProd(*[CategoricalFix(x["theta"]["nu"][0]) for _ in range(n)]),
        dim=n,
    )
    prior = StructDist(prior_dict)
    return prior


def define_VEM_base_dist(tau_tilde, e_tilde, mean_gamma, cov_gamma):
    """Define the base distribution for the VEM algorithm results."""
    base_dist_dict = OrderedDict()
    base_dist_dict["theta"] = StructDist(
        {
            "gamma": MvNormal(loc=mean_gamma, cov=cov_gamma),
            "nu": Dirichlet(e_tilde),
        }
    )  # scale use as a relaxation parameter (prior too tight)
    n = len(tau_tilde)
    base_dist_dict["Z"] = Cond(
        lambda x: IndepProd(*[CategoricalFix(tau_tilde[i]) for i in range(n)]), dim=n
    )  # CategoricalFix
    base_dist = StructDist(base_dist_dict)
    return base_dist
