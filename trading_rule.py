from get_data import *
from deap import gp, base, creator, tools
import random
import operator
import ta

count = 1
ohlcv = get_OHLCV()
roc_5 = ta.momentum.roc(ohlcv['Close'], window=5)
roc_10 = ta.momentum.roc(ohlcv['Close'], window=10)
williams_5 = ta.momentum.williams_r(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], lbp=5)
williams_10 = ta.momentum.williams_r(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], lbp=10)
kama_5 = ta.momentum.kama(ohlcv['Close'], window=5, pow1=2, pow2=30)
kama_10 = ta.momentum.kama(ohlcv['Close'], window=10, pow1=2, pow2=30)
atr_5 = ta.volatility.average_true_range(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], window=5)
atr_10 = ta.volatility.average_true_range(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], window=10)

class Bool:
    TRUE = True
    FALSE = False

def volume():
    volume = ohlcv['Volume']
    return volume

def calc(x: pd.Series, y: int) -> float:
    if y >= len(x):
        y = len(x) - 1
    if len(x.shape) == 1:
        return x.loc[y]
    else:
        return x.iloc[y, "Close"]
    
def num():
    return 1

def window():
    return 200
# Define the evaluation function that maps a trading rule tree to a fitness value
def evaluate(buy_func, sell_func):
    # to view the individual number (for debugging)
    global count
    count += 1
    # Convert the individual to a callable function
    buy = gp.compile(buy_func, pset)
    sell = gp.compile(sell_func, pset)
    
    btc_balance = 0
    aud_balance = 100

    for i in range(len(ohlcv)):
        buy_signal = buy(i, roc_5, roc_10, williams_5, williams_10, kama_5, kama_10, atr_5, atr_10)
        sell_signal = sell(i, roc_5, roc_10, williams_5, williams_10, kama_5, kama_10, atr_5, atr_10)
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

    if aud_balance > 60 and aud_balance != 100:
        print("individual number: ", count, "    aud balance: ", aud_balance)
    if aud_balance > 100:
        print("buy function: ", buy_func, "    sell function: ", sell_func)

    return aud_balance



# A class to represent the fitness of an individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# A class to represent an individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initialize the primitive set
pset = gp.PrimitiveSetTyped("main", [int, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series], Bool)
# Define the functions that can be used in the tree
pset.addPrimitive(calc, [pd.Series, int], float)
pset.addPrimitive(num, [], int)
pset.addPrimitive(volume, [], pd.Series)

pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(operator.mul, [int, float], float)
pset.addPrimitive(operator.and_, [Bool, Bool], Bool)
pset.addPrimitive(operator.or_, [Bool, Bool], Bool)
pset.addPrimitive(operator.not_, [Bool], Bool)
pset.addPrimitive(operator.gt, [float, float], Bool)

pset.renameArguments(ARG0="index")
pset.renameArguments(ARG1="roc_5"); pset.renameArguments(ARG2="roc_10")
pset.renameArguments(ARG3="williams_5"); pset.renameArguments(ARG4="williams_10")
pset.renameArguments(ARG5="kama_5"); pset.renameArguments(ARG6="kama_10")
pset.renameArguments(ARG7="atr_5"); pset.renameArguments(ARG8="atr_10")

#  Define the terminals that can be used in the tree
pset.addTerminal(0.1, float); pset.addTerminal(0.2, float); pset.addTerminal(0.3, float); pset.addTerminal(0.4, float); pset.addTerminal(0.5, float)
pset.addTerminal(False, Bool)
pset.addTerminal(True, Bool)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1), float) 

# Initialize the toolbox
toolbox = base.Toolbox()
# a generator function that generates trees of functions and operands using the primitive set pset.
toolbox.register(
    "expr", gp.genHalfAndHalf, pset=pset, min_= 3, max_= 5
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
toolbox.register("select", tools.selTournament, tournsize=4)
# a crossover operator that applies one-point crossover to two individuals.
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_= 3, max_= 5)
# a mutation operator that applies uniform mutation to an individual.
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

pop_size = 1000
CXPB, MUTPB, NGEN = 0.8, 0.2, 30
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
    print("------------------------ Generation %i --------------------------" % g)
    
    # decrease population size to remove bad individuals
    if (len(pop_buy) > 100):
        newPop = len(pop_buy) - 50
    else:
        newPop = len(pop_buy)

    print("Population size: ", newPop)

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

final = evaluate(best_buy, best_sell)
print("Final fitness: ", final)

