from sga import sgaAlgorithm
import configparser
import array
import random
import json
import numpy
from math import sqrt
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import matplotlib.pyplot as plt

# region config loading

config = configparser.ConfigParser()
config.read("config.ini")

BOUND_LOW = float(config['Values']['BOUND_LOW'])
BOUND_UP = float(config['Values']['BOUND_UP'])
NDIM = int(config['Values']['NDIM'])
CXPB = float(config['Values']['CXPB'])
MUTPB = float(config['Values']['MUTPB'])
NGEN = int(config['Values']['NGEN'])
NPOPULATION = int(config['Values']['NPOPULATION'])
ALGORITHM = int(config['Values']['ALGORITHM'])
MU = int(config['Values']['MU'])
LAMBDA = int(config['Values']['LAMBDA'])
MATING = config['Values']['MATING']
MUTATION = config['Values']['MUTATION']
EVAL_FUNC = config['Values']['EVAL']

# endregion


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='d',
               fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

if MATING == 'TwoPoint':
    toolbox.register("mate", tools.cxTwoPoint)
else:
    toolbox.register("mate", tools.cxOnePoint)

if MUTATION == 'gaussian':
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.7, indpb=MUTPB)
else:
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUTPB)

toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", eval(EVAL_FUNC))


def main():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    halloffame = tools.HallOfFame(maxsize=1)
    population = toolbox.population(n=NPOPULATION)
    if ALGORITHM == 0:
        population, logbook = sgaAlgorithm(
            population, toolbox, NGEN, CXPB, MUTPB, halloffame, stats)
    else:
        population, logbook = algorithms.eaMuCommaLambda(
            population, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame)

    return population, halloffame, logbook


if __name__ == "__main__":

    population, halloffame, logbook = main()

    print(halloffame)

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    size_avgs = logbook.select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()
