import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph

model_forms = {'sin':'sin(X_0)'}

def standard_model(model='sin'):

    model = AGraph(equation=model_forms[model])
    
    return model


