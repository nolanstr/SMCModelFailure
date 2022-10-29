import numpy as np

def get_galileo_data(with_shelf=True):

    if with_shelf:
        D = np.array([800, 1172, 1328, 1340, 1500]).reshape((-1,1))
        H = np.array([300, 600, 800, 828, 1000]).reshape((-1,1))
    
    else:
        D = np.array([253, 337, 395, 451, 495, 534, 573]).reshape((-1,1))
        H = np.array([100, 200, 300, 450, 600, 800, 1000]).reshape((-1,1))

    galileo_dict = {"D":D, "H":H}

    return galileo_dict

