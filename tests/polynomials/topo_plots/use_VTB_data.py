import numpy as np
import matplotlib.pyplot as plt

import sys;sys.path.append('../../../')
from models.polynomial import polynomial
from models.standard_model import standard_model
from data.generate_data import generate_data
from fitness.fitness import deterministic, new_bayes

sys.path.append('../../../../SMCBingoApplicationPaper/')
from util.data_functions.get_data import *

def make_training_data():
    
    data, num_points = organize_data([1,2,3])
    
    cols = np.append([0], np.arange(2, data.shape[1]))
    x, y = data[:,cols], data[:,1]
    training_data = ExplicitTrainingData(x, y)
    
    return training_data, num_points

models = [polynomial(n, input_dim=3) for n in range(10)]
import pdb;pdb.set_trace()

training_data, num_pts = make_training_data()

variances = np.zeros((x_sizes.shape[0], len(models), n_reevals))

for i, x_size in enumerate(x_sizes):

    x = np.random.uniform(low=0, high=np.pi, size=x_size).reshape((-1,1))
    x, y, y_noisy = generate_data(true_model, x, std=1, return_noisy=True)

    clo = deterministic(x, y)
    bff = new_bayes(clo)
    
    for j, model in enumerate(models):
        model_var = np.empty(n_reevals)

        for k in range(n_reevals):
            model._needs_opt = True
            clo(model)
            f = model.evaluate_equation_at(x)
            p = model.get_number_local_optimization_params()
            ssqe = np.sum((f - y) ** 2)
            #var_ols = ssqe / max(1, len(f) - p)
            var_ols = ssqe / len(f) 
            model_var[k] = var_ols
        variances[i, j, :] = model_var

mean_variances = np.nanmean(variances, axis=2)
X, Y = np.meshgrid(np.arange(len(models)), x_sizes)

plt.contourf(X, Y, mean_variances, 1000)
plt.colorbar()
plt.xlabel('Polynomial order')
plt.ylabel('number of datapoints')
plt.show()

import pdb;pdb.set_trace()
