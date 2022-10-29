import numpy as np
import matplotlib.pyplot as plt

import sys;sys.path.append('../../../')
from models.polynomial import polynomial
from models.standard_model import standard_model
from data.generate_data import generate_data
from data.galileo_data import get_galileo_data
from fitness.fitness import deterministic, new_bayes

models = [polynomial(n) for n in range(10)]

galileo_data = get_galileo_data(False)
x, y = galileo_data['H'], galileo_data['D']

x_sizes = np.arange(1, x.shape[0])
n_reevals = 1

variances = np.zeros((x_sizes.shape[0], len(models), n_reevals))

for i, x_size in enumerate(x_sizes):
    x_i, y_i = x[:x_size,:], y[:x_size,:]
    clo = deterministic(x_i, y_i)
    bff = new_bayes(clo)

    for j, model in enumerate(models):

        model_var = np.empty(n_reevals)
        
        for k in range(n_reevals):
            model._needs_opt = True
            clo(model)
            f = model.evaluate_equation_at(x_i)
            p = model.get_number_local_optimization_params()
            ssqe = np.sum((f - y_i) ** 2)
            var_ols = ssqe / (len(f) - p)
            #var_ols = ssqe / len(f) 
            model_var[k] = var_ols
            #import pdb;pdb.set_trace()
        variances[i, j, :] = model_var

mean_variances = np.nanmean(variances, axis=2)
X, Y = np.meshgrid(np.arange(len(models)), x_sizes)

cont = plt.contourf(X, Y, mean_variances)#, 1000, vmin=-1, vmax=1e4)
cbar = plt.colorbar()
plt.xlabel('polynomial order')
plt.ylabel('number of datapoints')
plt.plot(x_sizes, x_sizes, 'k', label='nan/inf line')
plt.title("var_ols for polynomials with Galileo's data (no shelf)")
plt.legend()
plt.savefig("var_ols_no_shelf", dpi=1000)
plt.show()

mean_variances[np.where(mean_variances<0)] = -1

import pdb;pdb.set_trace()


cont = plt.contourf(X, Y, mean_variances)#, 1000, vmin=-1, vmax=1e4)
cbar = plt.colorbar()
plt.xlabel('polynomial order')
plt.ylabel('number of datapoints')
plt.plot(x_sizes, x_sizes, 'k', label='nan/inf line')
plt.title("var_ols for polynomials with Galileo's data (no shelf)")
plt.legend()
plt.savefig("var_ols_no_shelf_with_fix", dpi=1000)
plt.show()

import pdb;pdb.set_trace()
