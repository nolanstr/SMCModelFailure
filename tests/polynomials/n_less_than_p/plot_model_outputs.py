import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


f = open("relevant_info.pkl", "rb")
data = pickle.load(f)
f.close()

x_instances = data['x_instances']
tags = data['model tags']
models = data['models']
x, y_data = data['x'], data['y data']
bffs = data['bffs']
import pdb;pdb.set_trace()
COLORS = [plt.cm.tab20c(4*i) for i in range(len(tags))]

pp = PdfPages('model_outputs.pdf')

for i, x in enumerate(x_instances):
    x_lin = np.linspace(x.min(), x.max(), 200).reshape((200,1))
    y = y_data[i] 
    bff = bffs[i]
    import pdb;pdb.set_trace()
    fig, axis = plt.subplots()
    for j, model in enumerate(models):
        model.set_local_optimization_params([0.]*model.get_number_local_optimization_params()) 
        axis.plot(x_lin.flatten(), model.evaluate_equation_at(x_lin).flatten(),
                                                            c=COLORS[j],
                                                            label=tags[j])
        nmll, step_list, _ = bff(model, return_nmll_only=False)
        if nmll == None:
            x, cred_y, pred_y = bff.get_cred_pred(copy.deepcopy(model), step_list)
            axis.fill_between(x.flatten(),
                              pred_y[:,0].flatten(),
                              pred_y[:,1].flatten(),
                              color=COLORS[j],
                              alpha=0.2)
            axis.fill_between(x.flatten(),
                              cred_y[:,0].flatten(),
                              cred_y[:,1].flatten(),
                              color=COLORS[j],
                              alpha=0.7)
            
        else:
            print(f'model {tags[j]} failed')

    axis.scatter(x.flatten(), y.flatten(), c='k', alpha=0.35, s=8)
    axis.set_ylim(bottom=min(0.5*y.min(), 1.5*y.min()), 
                  top=max(0.5*y.max(), 1.5*y.max()))

    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_title(f'x length = {len(x)}')
    axis.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    pp.savefig(fig, dpi=1000, transparent=True)
        

pp.close()        
        

