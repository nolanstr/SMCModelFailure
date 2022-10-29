from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy


def cred2din3d(x, cred, z, color):
    col_obj = ax.fill_between(x, cred[:,0].flatten(), cred[:,1].flatten(),
                                                    step='pre', alpha=0.1,
                                                    color=color) 
    ax.add_collection3d(col_obj, zs = z, zdir = 'z')

def plot2din3d(model_x, model_y, z, color):
    ax.plot(model_x, model_y, zs=z, zdir='zV', c=color)
    

def scatter2din3d(x,y,z):
    col_obj = ax.scatter(x, y, c='k') 
    ax.add_collection3d(col_obj, zs=z, zdir = 'z')

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


for i, sigma in enumerate(sigmas):
    y = y_data[i] 
    bff = bffs[i]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for j, model in enumerate(models):
        
        model_y = model.evaluate_equation_at(x_lin).flatten()
        nmll, step_list, _ = bff(model, return_nmll_only=False)
        x_new, cred_y, pred_y = bff.get_cred_pred(copy.deepcopy(model), step_list)
        
        plot2din3d(x_lin.flatten(), model_y, j, COLORS[j])
        cred2din3d(x_new.flatten(), pred_y, j, COLORS[j])
    scatter2din3d(x.flatten(), y.flatten(), 0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('number of terms (true model = 0)')
    plt.show()
