import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


f = open("relevant_info.pkl", "rb")
data = pickle.load(f)
f.close()

sigmas = data['sigmas']
tags = data['model tags']
models = data['models']
x, y_data = data['x'], data['y data']
x_lin = np.linspace(x.min(), x.max(), 200).reshape((200,1))
bffs = data['bffs']
COLORS = [plt.cm.tab20c(4*i) for i in range(len(tags))]

pp = PdfPages('model_outputs.pdf')

for i, sigma in enumerate(sigmas):
    y = y_data[i] 
    bff = bffs[i]
    fig, axis = plt.subplots()
    for j, model in enumerate(models):
        
        axis.plot(x_lin.flatten(), model.evaluate_equation_at(x_lin).flatten(),
                                                            c=COLORS[j],
                                                            label=tags[j])
        nmll, step_list, _ = bff(model, return_nmll_only=False)
        x, cred_y, pred_y = bff.get_cred_pred(copy.deepcopy(model), step_list)
        axis.fill_between(x.flatten(),
                          pred_y[:,0].flatten(),
                          pred_y[:,1].flatten(),
                          color=COLORS[j],
                          alpha=0.2)
        #axis.fill_between(x.flatten(),
        #                  cred_y[:,0].flatten(),
        #                  cred_y[:,1].flatten(),
        #                  color=COLORS[j],
        #                  alpha=0.7)

    axis.scatter(x.flatten(), y.flatten(), c='k', alpha=0.35, s=8)
    axis.set_ylim(bottom=min(0.5*y.min(), 1.5*y.min()), 
                  top=max(0.5*y.max(), 1.5*y.max()))

    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_title(f'std = {sigma}')
    axis.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    pp.savefig(fig, dpi=1000, transparent=True)
        

pp.close()        
        

