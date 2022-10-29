import numpy as np
import pickle

from smcpy import AdaptiveSampler
from smcpy import MultiSourceNormal
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel

import sys;sys.path.append('../../')
from data.generate_data import generate_data
from models.polynomial import polynomial
from fitness.fitness import deterministic, new_bayes, ensemble

n, p = 15, 3
ns = 0.01
multisource_num_pts = tuple([n])

model = polynomial(p)
model.set_local_optimization_params([2.]*(p+1))
x = np.linspace(1, 2, n).reshape((-1,1))

x, y, y_noisy = generate_data(model, x, std=1, return_noisy=True)
clo = deterministic(x, y_noisy)
bff = new_bayes(clo)

particles, mcmc_steps, ess_threshold = 600, 15, 0.75
norm_phi = 1 / np.sqrt(n)

clo(model)
param_names, priors = bff._create_priors(model, multisource_num_pts, particles)

print(str(model))
proposal = bff.generate_proposal_samples(model, particles, param_names)
log_like_args = [multisource_num_pts, tuple([None]*len(multisource_num_pts))]
log_like_func = MultiSourceNormal
vector_mcmc = VectorMCMC(lambda x: bff.evaluate_model(x, model),
                         y_noisy.flatten(),
                         priors,
                         log_like_args,
                         log_like_func)

mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=param_names)
smc = AdaptiveSampler(mcmc_kernel)

step_list, marginal_log_likes = smc.sample(particles,
                                           mcmc_steps,
                                           ess_threshold,
                                           proposal=proposal,
                                           required_phi=norm_phi)
nmll = -1 * (marginal_log_likes[-1] -
                     marginal_log_likes[smc.req_phi_index[0]])


import pdb;pdb.set_trace()
