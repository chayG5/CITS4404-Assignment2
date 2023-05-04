from get_data import *
from deap import gp, base, creator, tools
import random
import operator
import ta
import numpy as np

count = 0
ohlcv = get_OHLCV()
size = 325

training_data = ohlcv.iloc[:size, :]
test_data = ohlcv.iloc[size:, :]
test_data.reset_index(drop=True, inplace=True)

test_close = test_data['Close']; test_volume = test_data['Volume']
train_close = training_data['Close']; train_volume = training_data['Volume']
test_high = test_data['High']; test_low = test_data['Low']
train_high = training_data['High']; train_low = training_data['Low']

rsi_test = ta.momentum.rsi(test_close, window=14)
rsi_train = ta.momentum.rsi(train_close, window=14)
macd_test = ta.trend.macd(test_close, 26, 12, 9)
macd_train = ta.trend.macd(train_close, 26, 12, 9)
macd_signal_test = ta.trend.macd_signal(test_close, 26, 12, 9)
macd_signal_train = ta.trend.macd_signal(train_close, 26, 12, 9)
bbh_test = ta.volatility.bollinger_hband(test_close, window=20, window_dev=2)
bbh_train = ta.volatility.bollinger_hband(train_close, window=20, window_dev=2)
obv_test = ta.volume.on_balance_volume(test_close, test_volume)
obv_train = ta.volume.on_balance_volume(train_close, train_volume)
stoch_test = ta.momentum.stoch(test_close, test_high, test_low, 14, 3)
stoch_train = ta.momentum.stoch(train_close, train_high, train_low, 14, 3)
sma_20_test = ta.trend.sma_indicator(test_close, window=20)
sma_20_train = ta.trend.sma_indicator(train_close, window=20)
sma_50_test = ta.trend.sma_indicator(test_close, window=50)
sma_50_train = ta.trend.sma_indicator(train_close, window=50)

class Bool:
    TRUE = True
    FALSE = False

class Constant:
    one = 0.1
    two = 0.2
    three = 0.3
    four = 0.4
    five = 0.5
    actualOne = 1

def comparemacd(t:int) -> Bool:
    if t >= size:
        return macd_test.loc[t] > macd_signal_test.loc[t]
    else:
        return macd_train.loc[t] > macd_signal_train.loc[t]

def rsi_30(t: int) -> Bool:
    if t >= size:
        return rsi_test.loc[t] < 30
    else:
        return rsi_train.loc[t] < 30

def rsi_70(t: int) -> Bool:
    if t >= size:
        return rsi_test.loc[t] > 70
    else:
        return rsi_train.loc[t] > 70


def detectbbh(t: int) -> Bool:
    if t >= size:
        return test_close.loc[t] > bbh_test.loc[t]
    else:
        return train_close.loc[t] > bbh_train.loc[t]

def detectObv(t: int) -> Bool:
    if t == 0:
        return False
    if t >= size:
        return obv_test.loc[t] > obv_test.loc[t-1]
    else:
        return obv_train.loc[t] > obv_train.loc[t-1]

def Stoch_20(t: int) -> Bool:
    if t >= size:
        return stoch_test.loc[t] < 20
    else:
        return stoch_train.loc[t] < 20

def Stoch_80(t: int) -> Bool:
    if t >= size:
        return stoch_test.loc[t] > 80
    else:
        return stoch_train.loc[t] > 80

def sma_20_50(t: int) -> Bool:
    if t >= size:
        return sma_20_test.loc[t] > sma_50_test.loc[t]
    else:
        return sma_20_train.loc[t] > sma_50_train.loc[t]

def sma_50_20(t: int) -> Bool:
    if t >= size:
        return sma_20_test.loc[t] < sma_50_test.loc[t]
    else:
        return sma_20_train.loc[t] < sma_50_train.loc[t]

def num():
    return 1

# Define the evaluation function that maps a trading rule tree to a fitness value
def evaluate(buy_func, sell_func, data):
    # to view the individual number (for debugging)
    global count
    count += 1
    # Convert the individual to a callable function
    buy = gp.compile(buy_func, pset)
    sell = gp.compile(sell_func, pset)
    
    btc_balance = 0
    aud_balance = 100

    for i in range(len(data)):
        buy_signal = buy(i)
        sell_signal = sell(i)
        if buy_signal and not sell_signal and aud_balance > 0:
            aud_balance = 0.98*aud_balance
            btc_balance = aud_balance / data.loc[i, "Close"]
            aud_balance = 0
        elif sell_signal and not buy_signal and btc_balance > 0:
            aud_balance = btc_balance * data.loc[i, "Close"]
            btc_balance = 0
            aud_balance = 0.98*aud_balance

    # sell any remaining BTC
    if btc_balance > 0:
        aud_balance = btc_balance * data.loc[i, "Close"]
        btc_balance = 0
        aud_balance = 0.98*aud_balance

    if aud_balance > 60 and aud_balance != 100:
        print("individual number: ", count, "    aud balance: ", aud_balance)
        # print("buy function: ", buy_func, "    sell function: ", sell_func)
    if aud_balance > 100:
        print("buy function: ", buy_func, "    sell function: ", sell_func)

    return aud_balance

# A class to represent the fitness of an individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# A class to represent an individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initialize the primitive set
pset = gp.PrimitiveSetTyped("main", [int], Bool)
# Define the functions that can be used in the tree

pset.addPrimitive(num, [], int)
# pset.addPrimitive(cons, [], Constant)
# pset.addPrimitive(volume, [], pd.Series)

pset.addPrimitive(comparemacd, [int], Bool)
pset.addPrimitive(rsi_30, [int], Bool)
pset.addPrimitive(rsi_70, [int], Bool)
pset.addPrimitive(detectbbh, [int], Bool)
pset.addPrimitive(Stoch_20, [int], Bool)
pset.addPrimitive(Stoch_80, [int], Bool)
pset.addPrimitive(detectObv, [int], Bool)
pset.addPrimitive(sma_20_50, [int], Bool)
pset.addPrimitive(sma_50_20, [int], Bool)

pset.addPrimitive(operator.and_, [Bool, Bool], Bool)
pset.addPrimitive(operator.or_, [Bool, Bool], Bool)
pset.addPrimitive(operator.not_, [Bool], Bool)

pset.renameArguments(ARG0="index")



#  Define the terminals that can be used in the tree
pset.addTerminal(False, Bool)
pset.addTerminal(True, Bool)


# Initialize the toolbox
toolbox = base.Toolbox()
# a generator function that generates trees of functions and operands using the primitive set pset.
toolbox.register(
    "expr", gp.genHalfAndHalf, pset=pset, min_= 2, max_= 4
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
toolbox.register("expr_mut", gp.genFull, min_= 2, max_= 4)
# a mutation operator that applies uniform mutation to an individual.
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

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

    x_gen = []
    y_profits = []
    y_avgProfit = []

    # for plotting
    x_gen.append(0)
    y_profits.append((np.sum(np.array(fitnesses) > 100)/pop_size)*100)
    y_avgProfit.append(np.mean((np.array(fitnesses) - 100)))

    # run the genetic algorithm
    for g in range(NGEN):
        print("------------------------ Generation %i --------------------------" % g)
        x_gen.append(g+1)
        # decrease population size to remove bad individuals
        if (len(pop_buy) > 500):
            newPop = len(pop_buy) - 300
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

        all_fitnesses = [ind_buy.fitness.values[0] for ind_buy in pop_buy]
        y_profits.append((np.sum(np.array(all_fitnesses) > 100)/newPop)*100)
        y_avgProfit.append(np.mean((np.array(all_fitnesses) - 100)))

    return pop_buy, pop_sell, x_gen, y_avgProfit

