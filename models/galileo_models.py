import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph

def galileo_models(with_shelf=True):
    
    if with_shelf:
        galileo_string = "1.0 * sqrt(X_0)"
        print("f(H) = D")
    else:
        galileo_string = "(1.0 * sqrt(X_0)) / (1 + 1.0*X_0)"
        print("f(D) = H")

    model = AGraph(equation=galileo_string)
    
    return model
