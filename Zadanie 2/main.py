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

from pymop.problems.zdt import ZDT1

from nsga2spea2 import nsgaSpea
from nsga3 import nsga3

# region config loading

config = configparser.ConfigParser()
config.read("config.ini")

BOUND_LOW = float(config['Values']['BOUND_LOW'])
BOUND_UP = float(config['Values']['BOUND_UP'])
NDIM = int(config['Values']['NDIM'])
ALGORITHM = config['Values']['ALGORITHM']
NGEN = int(config['Values']['NGEN'])
MU = int(config['Values']['MU'])
CXPB = float(config['Values']['CXPB'])
MUTPB = float(config['Values']['MUTPB'])


# endregion


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d',
               fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

problem = ZDT1(n_var=NDIM)

toolbox.register("evaluate", problem.evaluate)

if ALGORITHM == 'NSGA3':
    mutateEta = 30.0
else:
    mutateEta = 20.0

toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=mutateEta)

toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)

pareto = problem.pareto_front(n_pareto_points=MU)


def main():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    halloffame = tools.HallOfFame(maxsize=1)
    # population = toolbox.population(n=NPOPULATION)
    if ALGORITHM == 'NSGA2':
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("tournament", tools.selTournamentDCD)
        population, logbook = nsgaSpea(
            NGEN, MU, CXPB, toolbox, halloffame, stats)
    elif ALGORITHM == 'SPEA2':
        toolbox.register("select", tools.selSPEA2)
        # https://books.google.pl/books?id=vpaNfpk79SgC&pg=PA269&lpg=PA269&dq=spea2+tournament&source=bl&ots=OKhBSF7BRI&sig=ACfU3U0lv1CDxCtJe0CAan7GY1zR_WyH-A&hl=pl&sa=X&ved=2ahUKEwjF19yPwp3pAhXGtYsKHd0cDXoQ6AEwAXoECAoQAQ#v=onepage&q=spea2%20tournament&f=false
        toolbox.register("tournament", tools.selTournament, tournsize=2)
        population, logbook = nsgaSpea(
            NGEN, MU, CXPB, toolbox, halloffame, stats)
    elif ALGORITHM == 'NSGA3':
        toolbox.register('select', tools.selNSGA3,
                         ref_points=pareto)
        population, logbook = nsga3(
            NGEN, MU, CXPB, MUTPB, toolbox, halloffame, stats)

    return population, halloffame, logbook


if __name__ == "__main__":

    population, halloffame, logbook = main()

    pop = numpy.array([ind.fitness.values for ind in population])

    population.sort(key=lambda x: x.fitness.values)

    print(halloffame)
    # print(logbook)
    print('DIVERSITY')
    print(diversity(population, pareto[0], pareto[-1]))
    print('CONVERGENCE')
    print(convergence(population, pareto))

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.scatter(pareto[:, 0], pareto[:, 1], color="blue", label="Pareto")
    ax1.scatter(pop[:, 0], pop[:, 1], color="red",
                label="Population", marker="x")

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(ALGORITHM)
    plt.legend()
    plt.show()
