import random
from deap import tools


def sgaAlgorithm(popuplation, toolbox, ngen, cxpb, mutpb, halloffame, stats):
    pop = popuplation
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "avg", "std", "min",  "max"

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    for g in range(ngen):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        halloffame.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook
