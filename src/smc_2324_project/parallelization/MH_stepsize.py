import numpy as np
from matplotlib import pyplot as plt

import os
import sys

from src.smc_2324_project.simulate.generate_dataset import *
from src.smc_2324_project.tempering.base_dist import *
from src.smc_2324_project.tempering.tempering import *

from src.smc_2324_project.variational.posterior_gamma import hessian, sample_from_gamma
from src.smc_2324_project.variational.posterior_nu import sample_from_nu
from src.smc_2324_project.variational.SBM_regression import VEM

# retrieve args
args = sys.argv[1:]
MH_stepsize_factor = float(args[0])
num_particles = 1000
tau_1_exp_stepsize, tau_2_resampling = 0.9, 0.8

# parameters
n, k, p = 40, 2, 4
gamma_0 = np.array([1, 0, 3, 1.1, 2.2, 0.1, -0.3])
V_0 = np.eye(len(gamma_0))
e_0 = 3 * np.ones(k)

# prior
theta_prior = define_theta_prior(gamma_0, V_0, e_0)
prior = define_prior(theta_prior, n)

# generate dataset
alpha, beta, nu = generate_network_params(k, gamma_0, V_0, e_0)
X = generate_covariates(n, p)
sample_from_network((alpha, beta, nu), X)
theta = (alpha, beta, nu)
Z, Y = sample_from_network(theta, X, return_Z=True)

### VEM ###
particle = prior.rvs()[0]
gamma_init, nu_init = particle["theta"]["gamma"], particle["theta"]["nu"]
tau_init = np.array([nu for i in range(n)])
inferred_gamma, inferred_nu, inferred_tau = VEM(Y, X, gamma_init, nu_init, tau_init)
hess = hessian(Y, X, inferred_tau, inferred_gamma)

# posterior distribution
e_tilde = e_0 + np.sum(inferred_tau, axis=0)
invV0 = np.linalg.inv(V_0)
cov_gamma = np.linalg.inv((invV0 - hess))
mean_gamma = cov_gamma @ (invV0 @ gamma_0 - hess @ inferred_gamma)

# relaxation
relaxation = 0.1
cov_gamma = (1 - relaxation) * cov_gamma + relaxation * np.eye(len(gamma_0))
e_tilde = (1 - relaxation) * e_tilde + relaxation * e_0
inferred_nu = (1 - relaxation) * inferred_nu + relaxation * nu

# base_dist
base_dist = define_VEM_base_dist(inferred_tau, inferred_nu, mean_gamma, cov_gamma)

### SMC ###

# target
compute_llh_target = define_llh_target(prior, X, Y, k)

from particles.smc_samplers import TemperingBridge


class ToyBridge(TemperingBridge):
    def logtarget(self, theta):
        return compute_llh_target(theta)


toy_bridge = ToyBridge(base_dist=base_dist)

move = FixedLenMCMCSequence(
    mcmc=CustomGibbs(k, e_0, e_tilde, MH_stepsize_factor), len_chain=5
)

import particles

# finally
fk_tpr = FlexibeAdaptiveTempering(
    model=toy_bridge,
    len_chain=100,
    move=move,
    wastefree=False,
    tempering_step_size=1 - tau_1_exp_stepsize,
)  # tempering_step_size = 1-tau_1
alg = particles.SMC(
    fk=fk_tpr,
    N=num_particles,
    ESSrmin=tau_2_resampling,
    store_history=True,
    verbose=False,
)
alg.run()

# save
import pickle

_ = len(os.listdir("data/hyperparams_tuning"))
filename = "data/MH_stepsize/MH_stepsize_{}_{}.pkl".format(MH_stepsize_factor, _)
with open(filename, "wb") as f:
    pickle.dump(alg.hist, f)
