from get_data import *
from deap import gp, base, creator, tools, algorithms
import random
import operator
import ta
import numpy as np

count = 1
ohlcv = get_OHLCV()

def sma(window):
    if window == 0:
        window = 1
    sma = ta.trend.sma_indicator(ohlcv['Close'], window=window)
    return sma

def rsi(window):
    if window == 0:
        window = 1
    rsi = ta.momentum.rsi(ohlcv['Close'], window=window)
    return rsi

def calc(x: pd.Series, y: int) -> float:
    if y >= len(x):
        y = len(x) - 1
    if len(x.shape) == 1:
        return x.loc[y]
    else:
        return x.iloc[y, "Close"]

# Define the evaluation function that maps a trading rule tree to a fitness value
def evaluate(buy_func, sell_fuc):
    # to view the individual number (for debugging)
    global count
    count += 1
    # Convert the individual to a callable function
    buy = gp.compile(buy_func, pset)
    sell = gp.compile(sell_fuc, pset)
    
    btc_balance = 0
    aud_balance = 100

    for i in range(len(ohlcv)):
        buy_signal = buy(i, ohlcv["Close"])
        sell_signal = sell(i, ohlcv["Close"])
        # print("data index:", i)
        # print("buy signal: ", buy_signal)
        # print("sell signal: ", sell_signal)
        if buy_signal and not sell_signal and aud_balance > 0:
            aud_balance = 0.98*aud_balance
            btc_balance = aud_balance / ohlcv.loc[i, "Close"]
            aud_balance = 0
        elif sell_signal and not buy_signal and btc_balance > 0:
            aud_balance = btc_balance * ohlcv.loc[i, "Close"]
            btc_balance = 0
            aud_balance = 0.98*aud_balance

    # sell any remaining BTC
    if btc_balance > 0:
        aud_balance = btc_balance * ohlcv.loc[i, "Close"]
        btc_balance = 0
        aud_balance = 0.98*aud_balance
    if aud_balance > 58 and aud_balance != 100:
        print("individual number: ", count, "    aud balance: ", aud_balance)
    return aud_balance



# A class to represent the fitness of an individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# A class to represent an individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

class Bool(object): pass
# Initialize the primitive set
# pass the current price as an argument to the trading rule
pset = gp.PrimitiveSetTyped("main", [int, pd.Series], Bool)

# Define the functions that can be used in the tree
pset.addPrimitive(sma, [int], pd.Series)
pset.addPrimitive(rsi, [int], pd.Series)
# pset.addPrimitive(ta.volatility.bollinger_lband_indicator, [pd.Series, int], pd.Series)
# pset.addPrimitive(ta.volatility.bollinger_hband_indicator, [pd.Series, int], pd.Series)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(operator.mul, [int, float], float)
pset.addPrimitive(operator.mul, [int, int], int)
# pset.addPrimitive(num, [], int)
pset.addPrimitive(operator.and_, [Bool, Bool], Bool)
pset.addPrimitive(operator.or_, [Bool, Bool], Bool)
pset.addPrimitive(operator.not_, [Bool], Bool)
pset.addPrimitive(operator.gt, [float, float], Bool)
pset.addPrimitive(calc, [pd.Series, int], float)

pset.renameArguments(ARG0="index")


#  Define the terminals that can be used in the tree
pset.addTerminal(random.randint(1, 30), int, "window") 
pset.addTerminal(random.uniform(0, 1), float) 
pset.addTerminal(False, Bool)
pset.addTerminal(True, Bool)



pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1), float) # don't understand why this needed??

# Initialize the toolbox
toolbox = base.Toolbox()
# a generator function that generates trees of functions and operands using the primitive set pset.
toolbox.register(
    "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3
)  # min_ and max_ are the minimum and maximum heights of the generated trees.
# a function that creates a new individual from a generator function (expr).
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# a function that creates a population of individuals from a generator function (individual).
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# a function that compiles an individual into a callable function.
toolbox.register("compile", gp.compile, pset=pset)
# a function that evaluates the fitness of an individual.
toolbox.register("evaluate", evaluate)
# a selection operator that selects individuals using tournament selection.
toolbox.register("select", tools.selTournament, tournsize=3)
# a crossover operator that applies one-point crossover to two individuals.
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# a mutation operator that applies uniform mutation to an individual.
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

pop_size = 100
CXPB, MUTPB, NGEN = 0.5, 0.2, 50
# initialize populations for buy and sell functions separately
pop_buy = toolbox.population(n=pop_size)
pop_sell = toolbox.population(n=pop_size)
# evaluate fitness of initial populations (uses the evaluate function)
fitnesses = [toolbox.evaluate(ind_buy, ind_sell) for ind_buy, ind_sell in zip(pop_buy,pop_sell)]
# assign fitness values to the individuals
for ind_buy, fit, ind_sell in zip(
    pop_buy, fitnesses, pop_sell
):
    ind_buy.fitness.values = (fit,)
    ind_sell.fitness.values = (fit,)

# run the genetic algorithm
for g in range(NGEN):
    # select the parents
    if (len(pop_buy) > 30):
        newPop = len(pop_buy) - 5
    else:
        newPop = len(pop_buy)
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
        if random.random() < 0.5:
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
    fitnesses = [toolbox.evaluate(ind_buy, ind_sell) for ind_buy, ind_sell in zip(invalid_ind_buy,invalid_ind_sell)]
    
    for ind, fit in zip(invalid_ind_buy, fitnesses):
        ind.fitness.values = (fit,)

    for ind, fit in zip(invalid_ind_sell, fitnesses):
        ind.fitness.values = (fit,)

    # The population is entirely replaced by the offspring
    pop_buy[:] = offspring_buy
    pop_sell[:] = offspring_sell

# get the best buy and sell functions
best_buy = tools.selBest(pop_buy, k=1)[0]
best_sell = tools.selBest(pop_sell, k=1)[0]

# compile the best buy and sell functions
buy_func = gp.compile(best_buy, pset)
sell_func = gp.compile(best_sell, pset)
print(best_buy)
print(best_sell)



# # Run the genetic algorithm
# random.seed(0)
# pop = toolbox.population(n=25)
# hof = tools.HallOfFame(10)
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("avg", np.mean)
# stats.register("min", np.min)
# stats.register("max", np.max)
# pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
# best_strategy = hof[0]
