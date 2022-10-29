import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph

def polynomial(terms, input_dim=1):

    for n in range(terms+1):
                
        if n == 0:
            string = '1.0'
        else:
            for i in range(input_dim):
                string += f' + 1.0*(X_{i}**{n})'

    model = AGraph(equation=string)

    return model

