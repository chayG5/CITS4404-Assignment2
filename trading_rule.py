from get_data import *
from deap import gp, base, creator, tools, algorithms
import random
import operator
import ta
import numpy as np

count = 1

def sma(window):
    ohlcv = get_OHLCV()
    sma = ta.trend.sma_indicator(ohlcv['Close'], window=window)
    return sma

# def rsi(window):
#     rsi = ta.momentum.rsi_indicator(get_OHLCV()['Close'], window=window)
#     return rsi

# def num():
#     return 3


def calc(x: pd.Series, y: int) -> float:
    return x.iloc[y,0]

# Define the evaluation function that maps a trading rule tree to a fitness value
def evaluate(buy_func, sell_fuc):
    global count
    print("individual number: ", count)
    count += 1
    # Convert the individual to a callable function
    data = get_OHLCV()
    buy = gp.compile(buy_func, pset)
    sell = gp.compile(sell_fuc, pset)
    
    btc_balance = 0
    aud_balance = 100

    for i in range(len(data)):
        buy_signal = buy(i)
        sell_signal = sell(i)
        if buy_signal and not sell_signal:
            aud_balance = 0.98*aud_balance
            btc_balance = aud_balance / data.loc[i, "Close"]
            aud_balance = 0
        elif sell_signal and not buy_signal:
            aud_balance = btc_balance * data.loc[i, "Close"]
            btc_balance = 0
            aud_balance = 0.98*aud_balance

    # sell any remaining BTC at the end of the data period
    if btc_balance > 0:
        aud_balance = btc_balance * data.loc[i, "Close"]
        btc_balance = 0
        aud_balance = 0.98*aud_balance

    print("aud balance: ", aud_balance)
    return aud_balance



# A class to represent the fitness of an individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# A class to represent an individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initialize the primitive set
# pass the current price as an argument to the trading rule
pset = gp.PrimitiveSetTyped("main", [int], bool)

# Define the functions that can be used in the tree
pset.addPrimitive(sma, [int], pd.Series)
# pset.addPrimitive(ta.momentum.RSIIndicator, [pd.Series, int], pd.Series)
# pset.addPrimitive(ta.volatility.bollinger_lband_indicator, [pd.Series, int], pd.Series)
# pset.addPrimitive(ta.volatility.bollinger_hband_indicator, [pd.Series, int], pd.Series)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(operator.mul, [int, float], float)
pset.addPrimitive(operator.mul, [int, int], int)
# pset.addPrimitive(num, [], int)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
# pset.addPrimitive(calc, [pd.Series, int], float)


#  Define the terminals that can be used in the tree
pset.addTerminal(random.randint(1, 30), int) 
pset.addTerminal(random.uniform(0, 1), float) 
pset.addTerminal(0, bool)
pset.addTerminal(1, bool)
# pset.addTerminal(get_OHLCV()['Close'], pd.Series)



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

pop_size = 5
# initialize populations for buy and sell functions separately
pop_buy = toolbox.population(n=pop_size)
pop_sell = toolbox.population(n=pop_size)
# evaluate fitness of initial populations (uses the evaluate function)
fitnesses_buy = [toolbox.evaluate(ind, None) for ind in pop_buy]
fitnesses_sell = [toolbox.evaluate(None, ind) for ind in pop_sell]
# assign fitness values to the individuals
for ind_buy, fit_buy, ind_sell, fit_sell in zip(
    pop_buy, fitnesses_buy, pop_sell, fitnesses_sell
):
    ind_buy.fitness.values = fit_buy
    ind_sell.fitness.values = fit_sell

num_generations = 10
# run the genetic algorithm
for g in range(num_generations):
    # select the parents
    parents_buy = toolbox.select(pop_buy, len(pop_buy))
    parents_sell = toolbox.select(pop_sell, len(pop_sell))
    # create offspring using genetic operators
    offspring_buy = [toolbox.clone(ind) for ind in parents_buy]
    offspring_sell = [toolbox.clone(ind) for ind in parents_sell]
    # mutaion and crossover for buy and sell functions separately
    for child1, child2 in zip(offspring_buy[::2], offspring_buy[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
        toolbox.mutate(child1)
        toolbox.mutate(child2)
    for child1, child2 in zip(offspring_sell[::2], offspring_sell[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
        toolbox.mutate(child1)
        toolbox.mutate(child2)

    # evaluate the fitness of the offsprings made above
    fitnesses_buy = [toolbox.evaluate(ind, None) for ind in offspring_buy]
    fitnesses_sell = [toolbox.evaluate(None, ind) for ind in offspring_sell]
    # assign fitness values to the offspring
    for ind_buy, fit_buy, ind_sell, fit_sell in zip(
        offspring_buy, fitnesses_buy, offspring_sell, fitnesses_sell
    ):
        ind_buy.fitness.values = fit_buy
        ind_sell.fitness.values = fit_sell

    # select the new population
    pop_buy = toolbox.select(offspring_buy + pop_buy, k=pop_size)
    pop_sell = toolbox.select(offspring_sell + pop_sell, k=pop_size)

# get the best buy and sell functions
best_buy = tools.selBest(pop_buy, k=1)[0]
best_sell = tools.selBest(pop_sell, k=1)[0]

# compile the best buy and sell functions
buy_func = compile(best_buy, pset)
sell_func = compile(best_sell, pset)
print(buy_func)
print(sell_func)


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
