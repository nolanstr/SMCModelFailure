import numpy as np
import pickle

import sys;sys.path.append('../../../')
from data.generate_data import generate_data
from models.standard_model import standard_model
from models.taylors_series import taylors_series
from fitness.fitness import deterministic, bayes, ensemble

model = standard_model(model='sin')
N_terms = np.arange(1, 5, dtype=int)
taylor_models = [taylors_series(n, model='sin') for n in N_terms]
import pdb;pdb.set_trace()
x = np.linspace(0, np.pi, 25).reshape((-1,1))
size = 5
sigmas = np.power(np.linspace(0.01, 1.0, 10), 2) * 1.
fitness_estimates = np.empty((sigmas.shape[0], N_terms.shape[0]+1, size))
model_tags = ['true_model'] + [f'{n} terms' for n in N_terms]
clos, bffs, y_data = [], [], []

for i, sigma in enumerate(sigmas):

    x, y, y_noisy = generate_data(model, x, std=sigma, return_noisy=True)
    y_data.append(y_noisy)
    clo = deterministic(x, y_noisy)
    bff = bayes(clo)
    
    clos.append(clo), bffs.append(bff)

    true_model_fitness = ensemble(model, bff, size=size)
    fitness_estimates[i,0,:] = true_model_fitness
    print(f'true model nan median = {np.nanmedian(fitness_estimates[i,0,:])}')

    for j, taylor_model in enumerate(taylor_models):
        
        taylor_model_fitness = ensemble(taylor_model, bff, size=size)
        fitness_estimates[i,j+1,:] = taylor_model_fitness
        print(f'taylor model nan median = {np.nanmedian(fitness_estimates[i,j+1,:])}')


data = {'fitness':fitness_estimates, 
        'sigmas':sigmas,
        'models':[model]+taylor_models,
        'model tags':model_tags,
        'x':x,
        'y data':y_data,
        'clos':clos,
        'bffs':bffs}

f = open("relevant_info.pkl", "wb")
pickle.dump(data, f)
f.close()
import pdb;pdb.set_trace()
