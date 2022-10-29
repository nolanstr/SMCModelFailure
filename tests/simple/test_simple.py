import numpy as np
import pickle

import sys;sys.path.append('../../')
from data.generate_data import generate_data
from models.polynomial import polynomial
from fitness.fitness import deterministic, new_bayes, ensemble

N_terms = 6 
poly_model = polynomial(N_terms)

x = np.linspace(1, 2, 100).reshape((-1,1))
sigma = 0.1

x, y, y_noisy = generate_data(poly_model, x, std=0.1, return_noisy=True)
clo = deterministic(x, y_noisy)
bff = new_bayes(clo)

fitness = bff(poly_model)
print(f'true model -NMLL = {fitness}')

import pdb;pdb.set_trace()
