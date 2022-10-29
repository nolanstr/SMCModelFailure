import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph

model_forms = {'sin':{'num': lambda n: ' - ' if n%2==1 else ' + ',
                      'den': lambda n: str(np.math.factorial(((2*n)+1))),
                      'mult': lambda n: '(X_0**' + str(int(eval(f'2*{n}+1'))) + ')'}
              }

def taylors_series(terms, model='sin'):

    for n in range(terms):
        
        num = model_forms[model]['num'](n)
        den = model_forms[model]['den'](n)
        mult = model_forms[model]['mult'](n)
                
        if n == 0:
            n_expansion = '(((' + mult + ') / ' + den + '))'
            string = n_expansion
        else:
            n_expansion = '(((' + mult + ') / ' + den + '))'
            string += num + n_expansion
    print(string)
    model = AGraph(equation=string)
    print(str(model))
    return model

