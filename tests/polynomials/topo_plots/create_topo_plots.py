import numpy as np
import matplotlib.pyplot as plt

import sys;sys.path.append('../../../')
from models.polynomial import polynomial
from models.standard_model import standard_model
from data.generate_data import generate_data
from fitness.fitness import deterministic, new_bayes

models = [polynomial(n) for n in range(10)]
true_model = polynomial(10)

x_sizes = np.arange(5, 11)
n_reevals = 100

variances = np.zeros((x_sizes.shape[0], len(models), n_reevals))
x_total = np.random.uniform(low=0, high=np.pi, size=x_sizes.max()).reshape((-1,1))
x_total, y_total, y_noisy_total = generate_data(true_model, x_total, std=2.5, return_noisy=True)

for i, x_size in enumerate(x_sizes):

    x_noisy = np.random.choice(x_total.flatten(), x_size,
                                        replace=False).reshape((-1,1))
    y_noisy = np.random.choice(y_noisy_total.flatten(), x_size,
                                        replace=False).reshape((-1,1))
    clo = deterministic(x_noisy, y_noisy)
    bff = new_bayes(clo)
    
    for j, model in enumerate(models):
        model_var = np.empty(n_reevals)

        for k in range(n_reevals):
            model._needs_opt = True
            clo(model)
            f = model.evaluate_equation_at(x_noisy)
            p = model.get_number_local_optimization_params()
            ssqe = np.sum((f - y_noisy) ** 2)
            #var_ols = ssqe / (len(f) - p)
            var_ols = ssqe / len(f) 
            #print(ssqe)
            model_var[k] = var_ols
        variances[i, j, :] = model_var

mean_variances = np.nanmedian(variances, axis=2)
mean_variances[np.where(mean_variances<0)] = np.nan
mean_variances[np.isinf(abs(mean_variances))] = np.nan

X, Y = np.meshgrid(np.arange(len(models)), x_sizes)

plt.contourf(X, Y, mean_variances, linewidths=2)
plt.colorbar()
plt.xlabel('Polynomial order')
plt.ylabel('number of datapoints')
plt.show()

import pdb;pdb.set_trace()
