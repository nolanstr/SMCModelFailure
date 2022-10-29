import numpy as np
import pickle

import sys;sys.path.append('../../../')
from data.generate_data import generate_data
from models.polynomial import polynomial
from fitness.fitness import deterministic, bayes, ensemble

N_terms = np.arange(0, 4, dtype=int)
polynomials = [polynomial(n) for n in N_terms]

x_instances = [np.linspace(1, 2, x_n).reshape((-1,1)) for x_n in range(2,7)]
size = 5
sigma = 0.1
fitness_estimates = np.empty((len(x_instances), N_terms.shape[0]+1, size))
model_tags = ['true_model'] + [f'{n} terms' for n in N_terms]
print(model_tags)
import pdb;pdb.set_trace()
clos, bffs, y_data = [], [], []

for i, x in enumerate(x_instances):

    x, y, y_noisy = generate_data(polynomials[-1], x, std=0.1, return_noisy=True)
    y_data.append(y_noisy)
    clo = deterministic(x, y_noisy)
    bff = bayes(clo)
    print(bff.training_data.x.shape) 
    clos.append(clo), bffs.append(bff)

    true_model_fitness = ensemble(polynomials[-1], bff, size=size)
    fitness_estimates[i,0,:] = true_model_fitness
    print(f'true model nan median = {np.nanmedian(fitness_estimates[i,0,:])}')

    for j, polynomial_model in enumerate(polynomials):
        
        polynomial_fitness = ensemble(polynomial_model, bff, size=size)
        fitness_estimates[i,j+1,:] = polynomial_fitness 
        print(f'polynomial model nan median = {np.nanmedian(fitness_estimates[i,j+1,:])}')
        print(str(polynomial_model))

data = {'fitness':fitness_estimates, 
        'x_instances':x_instances,
        'models':[polynomials[-1]]+polynomials,
        'model tags':model_tags,
        'x':x,
        'y data':y_data,
        'clos':clos,
        'bffs':bffs}

f = open("relevant_info.pkl", "wb")
pickle.dump(data, f)
f.close()
import pdb;pdb.set_trace()
