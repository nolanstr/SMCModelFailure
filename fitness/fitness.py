import numpy as np

from bingo.symbolic_regression import ExplicitRegression, \
                                        ExplicitTrainingData
from bingo.local_optimizers.continuous_local_opt import \
                                        ContinuousLocalOptimization
from bingo.symbolic_regression.test_bayes_fitness_function import \
                                        BayesFitnessFunction 
from bingo.symbolic_regression.bayes_fitness.bayes_fitness_function import \
                                    BayesFitnessFunction as NewBayesFitnessFunction 

def deterministic(x, y):

    training_data = ExplicitTrainingData(x, y)
    fitness = ExplicitRegression(training_data=training_data)
    clo = ContinuousLocalOptimization(fitness, algorithm='lm')

    return clo

def bayes(clo, random_sample_info=None):

    smc_hyperparams = {'num_particles':600,
                       'mcmc_steps':15,
                       'ess_threshold':0.75}
    random_sample_info = random_sample_info

    bff = BayesFitnessFunction(clo,
                               smc_hyperparams=smc_hyperparams,
                               random_sample_info=random_sample_info)
    return bff

def new_bayes(clo, random_sample_info=None):

    smc_hyperparams = {'num_particles':600,
                       'mcmc_steps':15,
                       'ess_threshold':0.75}
    random_sample_info = random_sample_info

    bff = NewBayesFitnessFunction(clo,
                               smc_hyperparams=smc_hyperparams,
                               random_sample_info=random_sample_info)
    return bff

def ensemble(model, fitness, size=5):

    fitness_estimates = np.array([fitness(model) for _ in range(size)])

    return fitness_estimates
