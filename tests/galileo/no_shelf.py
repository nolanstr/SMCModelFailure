import numpy as np
import pickle

import sys;sys.path.append('../../')
from data.galileo_data import get_galileo_data
from models.galileo_models import galileo_models
from fitness.fitness import deterministic, new_bayes, ensemble

with_shelf = False

galileo_model = galileo_models(with_shelf)
galileo_data = get_galileo_data(with_shelf)

print(str(galileo_model))
x, y = galileo_data['H'], galileo_data['D']
clo = deterministic(x, y)
bff = new_bayes(clo)

fitness = bff(galileo_model)
print(f'true model -NMLL = {fitness}')

import pdb;pdb.set_trace()
