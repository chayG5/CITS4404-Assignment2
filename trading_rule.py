from get_data import *
from deap import gp, base, creator, tools, algorithms
import random
import operator
import ta
import numpy as np

count = 1

# def sma(window):
#     sma = ta.trend.sma_indicator(get_OHLCV()['Close'], window=window)
#     return sma

# def rsi(window):
#     rsi = ta.momentum.rsi_indicator(get_OHLCV()['Close'], window=window)
#     return rsi

def df_comparison(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    # Apply the comparison operator element-wise
    result = operator.gt(x, y)
    # Replace any NaN values with False
    result = result.replace(np.nan, False)
    return result

# Define the evaluation function that maps a trading rule tree to a fitness value
def evaluate(individual): #evaluate(buy, sell)
    global count
    print("ind" + str(count))
    count += 1
    # Convert the individual to a callable function
    strategy = gp.compile(individual, pset)
    
    # Simulate trades on historical data
    data = get_OHLCV()
    inputs = data['Close']
    output = strategy(inputs)
    # print(output.values)
    output = output.values
    balance = 1000
    for i in range(len(output)):
        if output[i] == 1:
            # Buy BTC with all available AUD balance
            balance -= 2
        elif output[i] == 0:
            # Sell all BTC
            balance += 2
    
    # Calculate the fitness (return on investment)
    roi = balance / 1000 - 1
    return roi,

def num():
    return 3

# A class to represent the fitness of an individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# A class to represent an individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initialize the primitive set
pset = gp.PrimitiveSetTyped("main", [pd.Series], pd.Series)

# [pd.Series, int], pd.Series
# Define the functions that can be used in the tree
pset.addPrimitive(ta.trend.sma_indicator, [pd.Series, int], pd.Series) 
# pset.addPrimitive(ta.momentum.RSIIndicator, [pd.Series, int], pd.Series)
pset.addPrimitive(ta.volatility.bollinger_lband_indicator, [pd.Series, int], pd.Series)
pset.addPrimitive(ta.volatility.bollinger_hband_indicator, [pd.Series, int], pd.Series)
pset.addPrimitive(df_comparison, [pd.Series, pd.Series], pd.DataFrame)
pset.addPrimitive(num,[],int)

# Add comparison operators
# pset.addPrimitive(operator.gt, arity=2)   # greater than
# pset.addPrimitive(operator.lt, arity=2)   # less than
# pset.addPrimitive(operator.eq, arity=2)   # equal to 
# # pset.addPrimitive(operator.and_, 2) # and
# # pset.addPrimitive(operator.or_, 2)  # or
# pset.addPrimitive(operator.not_, arity=2) # not
# pset.addPrimitive(operator.ge, arity=2)   # greater than
# pset.addPrimitive(operator.le, arity=2)   # less than


#  Define the terminals that can be used in the tree
pset.addTerminal(random.randint(5,20), int) # for the window size
# pset.addTerminal(random.choice[operator.le, operator.ge, operator.gt, operator.lt], callable)
# pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1)) # don't understand why this needed??

# Initialize the toolbox
toolbox = base.Toolbox()
# a generator function that generates trees of functions and operands using the primitive set pset.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) # min_ and max_ are the minimum and maximum heights of the generated trees.
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

# Run the genetic algorithm
random.seed(0)
pop = toolbox.population(n=25)
hof = tools.HallOfFame(10)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
best_strategy = hof[0]
