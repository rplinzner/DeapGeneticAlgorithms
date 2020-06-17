from deap import creator
from deap import tools
from deap import base
import numpy as np
import random

import core.helpers as helpers
from core.clusterData import ClusterData
from core.dataService import DataService

NGEN = 50
MUTPB = 0.2
CXPB = 0.8
MU = 60

numOfClusters = 3
numOfDataInCluster = 3

skipRows = 300
numberOfRows = 60

dataService = DataService(
    'D:\\GITHUB\\DeapGeneticAlgorithms\\Zadanie 3\\core\\3D_spatial_network.txt', numOfClusters, numberOfRows)

# SKIP first column
dataService.load_data([0], skipRows)

data = dataService.loadedData

clusterData = ClusterData(numOfClusters, numOfDataInCluster, data)

creator.create("Fitnesses", base.Fitness, weights=(-1.0, -1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.Fitnesses)

toolbox = base.Toolbox()

toolbox.register("attr_float", dataService.getRandomIndividual)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", clusterData.eval)
toolbox.register("mate", tools.cxUniform, indpb=0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.7, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("tournament", tools.selTournament, tournsize=2)


logbook = tools.Logbook()
logbook.header = "gen", "evals", "std", "min", "avg", "max"


def main(seed=None):
    random.seed(seed)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    halloffame = tools.HallOfFame(maxsize=1, similar=np.array_equal)
    pop = toolbox.population(n=MU)

   # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    halloffame.update(pop)
    print(halloffame)
    helpers.plotDataWithCentroids(
        np.array(halloffame[0]), numOfClusters, numOfDataInCluster, data)

    # Begin the generational process
    for gen in range(NGEN):
        # Vary the population
        offspring = toolbox.tournament(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, MU)
        halloffame.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook, halloffame


if __name__ == "__main__":
    pop, logbook, hof = main()
    print("FINAL:")
    print(hof)
    print(logbook.stream)

    helpers.plotDataWithCentroids(
        np.array(hof[0]), numOfClusters, numOfDataInCluster, data)
