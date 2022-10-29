import numpy as np

def generate_data(model, x, std=1, return_noisy=False):
    
    y = model.evaluate_equation_at(x)
    
    if return_noisy:
        y_noisy = y + np.random.normal(loc=0, scale=std, size=x.shape)
        return x, y, y_noisy

    else:
        return x, y


