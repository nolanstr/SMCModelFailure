import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys;sys.path.append('../../')
from data.generate_data import generate_data
from models.polynomial import polynomial
from models.standard_model import standard_model
from fitness.fitness import deterministic, new_bayes, ensemble

true_model = standard_model()
poly_model = polynomial(8)

def plot_on_pdf(pp, n):

    fig, axis = plt.subplots()

    x = np.linspace(1, 2, n).reshape((-1,1))
    x, y, y_noisy = generate_data(true_model, x, std=25, return_noisy=True)
    clo = deterministic(x, y_noisy)
    bff = new_bayes(clo)

    fitness = bff(poly_model)
    print(f'n = {n}, true model -NMLL = {fitness}')

    COLOR = plt.cm.tab20c(0)
    x_lin = np.linspace(x.min(), x.max(), 1000).reshape((-1,1))
    axis.plot(x_lin.flatten(), poly_model.evaluate_equation_at(x_lin).flatten(),
                                                        c=COLOR)
    nmll, step_list, _ = bff(poly_model, return_nmll_only=False)
    x, cred, pred = bff.estimate_cred_pred(copy.deepcopy(poly_model), step_list)
    axis.fill_between(x.flatten(),
                      pred[:,0].flatten(),
                      pred[:,1].flatten(),
                      color=COLOR,
                      alpha=0.2)
    axis.fill_between(x.flatten(),
                      cred[:,0].flatten(),
                      cred[:,1].flatten(),
                      color=COLOR,
                      alpha=0.7)
    
    axis.scatter(x.flatten(), y_noisy.flatten(), c='k', alpha=0.35, s=8)
    axis.set_ylim(bottom=min(0.5*y.min(), 1.5*y.min()), 
                  top=max(0.5*y.max(), 1.5*y.max()))

    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_title(f'n = {n}')
    plt.tight_layout()
    pp.savefig(fig, dpi=1000, transparent=True)


n_vals = [100,50,25,10,9,8]
pp = PdfPages('model_output.pdf')

for n in n_vals:
    plot_on_pdf(pp, n) 

pp.close()        
