import numpy as np
from matplotlib import pyplot as plt
import json

import os
import sys

os.chdir('./../../../..')
#print(os.getcwd())
sys.path.append(os.getcwd())

from src.smc_2324_project.simulate.generate_dataset import *
from src.smc_2324_project.tempering.base_dist import *
from src.smc_2324_project.tempering.tempering import *

import particles
from particles.smc_samplers import TemperingBridge
with open('mon_fichier.txt', 'w') as fichier:
    # Écriture du texte dans le fichier
    fichier.write('Bonjour')

class ToyBridge(TemperingBridge):
    def logtarget(self, theta):
        return compute_llh_target(theta)

# retrieve args
args = sys.argv[1:]
k_simu = int(args[0])
num_fichier = int(args[1])
num_particles = 200

# hyperparameters to generating the dataset
n, k, p = 40, 3, 4
gamma_0 = np.array([0.6,0.6,0.6,1, 0, 3, 1.1, 2.2, 0.1, -0.3])
V_0 = np.eye(len(gamma_0))
e_0 = 3 * np.ones(k)

# generate dataset
#the seed is fixed therefore all the simulations will be done with the same dataset
np.random.seed(42)
alpha, beta, nu = generate_network_params(k, gamma_0, V_0, e_0)
X = generate_covariates(n, p)
sample_from_network((alpha, beta, nu), X)
theta = (alpha,beta,nu)
Y = sample_from_network(theta,X, return_Z=False)


random.seed()



list_lpy=[]
k=k_simu
for i in range(5):
    #hyperparameters
    gamma_0 = np.random.normal(0, 1, (k * (k + 1)) // 2 + 4)
    V_0 = 3*np.eye(len(gamma_0))
    e_0 = 3 * np.ones(k)

    # prior
    theta_prior = define_theta_prior(gamma_0, V_0, e_0)
    prior = define_prior(theta_prior, n)
    
    #logtarget and model
    compute_llh_target = define_llh_target(prior, X, Y, k)
    toy_bridge = ToyBridge(base_dist=prior)
    
    #move
    move = FixedLenMCMCSequence(mcmc=CustomGibbs(k, e_0, e_0), len_chain=5)
    
    # adaptative tempering
    fk_tpr = FlexibeAdaptiveTempering(model=toy_bridge, len_chain=100, 
                                      move=move, wastefree=False,
                                      tempering_step_size = 0.5) # tempering_step_size = 1-tau_1
    alg = particles.SMC(fk=fk_tpr, N=2000, ESSrmin=0.8,
                        store_history=True, verbose=True)
    alg.run()
    
    #save data
    lpy=0
    for wgt in alg.hist.wgts:
        #lpy+=np.log(sum(np.exp(wgt.lw)))
        lpy+=max(wgt.lw) + np.log(sum(np.exp(wgt.lw-max(wgt.lw))))
    print(lpy)
    list_lpy.append(lpy)

# save
filename = "data\posterior_for_diff_k\posterior_k_{}_{}.json".format(k, num_fichier)
with open(filename, "w") as json_file:
    json.dump(list_lpy, json_file, indent=2)

