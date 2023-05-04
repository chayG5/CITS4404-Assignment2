import random
import numpy as np
from deap_initialisation import *


def genetic_program(pop, gen):
    pop_size = pop
    CXPB, MUTPB, NGEN = 0.5, 0.2, gen
    # initialize populations for buy and sell functions separately
    pop_buy = toolbox.population(n=pop_size)
    pop_sell = toolbox.population(n=pop_size)

    # evaluate fitness of initial populations (uses the evaluate function)
    fitnesses = [toolbox.evaluate(ind_buy, ind_sell, training_data) for ind_buy, ind_sell in zip(pop_buy,pop_sell)]
    # assign fitness values to the individuals
    for ind_buy, fit, ind_sell in zip(
        pop_buy, fitnesses, pop_sell
    ):
        ind_buy.fitness.values = (fit,)
        ind_sell.fitness.values = (fit,)

    # for plotting
    x_gen = []
    y_avgProfit = []

    # store the profits for plotting
    x_gen.append(0)
    y_avgProfit.append(np.mean((np.array(fitnesses) - 100)))

    # run the genetic algorithm
    for g in range(NGEN):
        g=g+1
        print("------------------------ Generation %i --------------------------" % g)
        x_gen.append(g)
        # decrease population size to remove bad individuals
        if (len(pop_buy) > 50):
            newPop = len(pop_buy) - 10
        else:
            newPop = len(pop_buy)

        # select the parents
        parents_buy = toolbox.select(pop_buy, newPop)
        parents_sell = toolbox.select(pop_sell, newPop)

        # create offspring using genetic operators
        offspring_buy = [toolbox.clone(ind) for ind in parents_buy]
        offspring_sell = [toolbox.clone(ind) for ind in parents_sell]

        # crossover for buy and sell functions separately
        for child1, child2 in zip(offspring_buy[::2], offspring_buy[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        for child1, child2 in zip(offspring_sell[::2], offspring_sell[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        # mutate the offspring
        for mutant in offspring_buy:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for mutant in offspring_sell:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind_buy = [ind for ind in offspring_buy if not ind.fitness.valid]
        invalid_ind_sell = [ind for ind in offspring_sell if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind_buy, ind_sell, training_data) for ind_buy, ind_sell in zip(invalid_ind_buy,invalid_ind_sell)]
        
        for ind, fit in zip(invalid_ind_buy, fitnesses):
            ind.fitness.values = (fit,)

        for ind, fit in zip(invalid_ind_sell, fitnesses):
            ind.fitness.values = (fit,)

        # The population is entirely replaced by the offspring
        pop_buy[:] = offspring_buy
        pop_sell[:] = offspring_sell

        # Get average profit for plottiing
        all_fitnesses = [ind_buy.fitness.values[0] for ind_buy in pop_buy]
        avg = np.mean((np.array(all_fitnesses) - 100))
        y_avgProfit.append(avg)
        print()
        print("Average Profit in this Generation: ", avg)
        print()

    return pop_buy, pop_sell, x_gen, y_avgProfit

