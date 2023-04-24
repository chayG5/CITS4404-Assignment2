from get_data import *
from deap import gp, base, creator, tools, algorithms
import random
import operator
import ta
import numpy as np



# def sma(window):
#     sma = ta.trend.sma_indicator(get_OHLCV()['Close'], window=window)
#     return sma

# def rsi(window):
#     rsi = ta.momentum.rsi_indicator(get_OHLCV()['Close'], window=window)
#     return rsi

# Define the evaluation function that maps a trading rule tree to a fitness value
def evaluate(individual):
    # Convert the individual to a callable function
    strategy = gp.compile(individual, pset)
    
    # Simulate trades on historical data
    data = get_OHLCV()
    balance = 1000
    for i in range(len(data)):
        inputs = data.iloc[i]['Close']
        output = strategy(inputs)
        if output == 1:
            # Buy BTC with all available AUD balance
            balance -= data.iloc[i]['Close'] * balance
        elif output == 0:
            # Sell all BTC
            balance += data.iloc[i]['Close']* balance
    
    # Calculate the fitness (return on investment)
    roi = balance / 1000 - 1
    return roi,


# A class to represent the fitness of an individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# A class to represent an individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initialize the primitive set
pset = gp.PrimitiveSetTyped("main", [float], bool)

# Define the functions that can be used in the tree
pset.addPrimitive(ta.trend.sma_indicator, [pd.Series, int], pd.Series) 
pset.addPrimitive(ta.momentum.RSIIndicator, [pd.Series, int], pd.Series)
pset.addPrimitive(ta.volatility.bollinger_lband_indicator, [pd.Series, int], pd.Series)
pset.addPrimitive(ta.volatility.bollinger_hband_indicator, [pd.Series, int], pd.Series)
# Add comparison operators
pset.addPrimitive(operator.gt, [pd.Series, pd.Series], bool)   # greater than
pset.addPrimitive(operator.lt, [pd.Series, pd.Series], bool)   # less than
pset.addPrimitive(operator.eq, [pd.Series, pd.Series], bool)   # equal to 
# pset.addPrimitive(operator.and_, 2) # and
# pset.addPrimitive(operator.or_, 2)  # or
pset.addPrimitive(operator.not_, [pd.Series, pd.Series], bool) # not
pset.addPrimitive(operator.ge, [pd.Series, pd.Series], bool)   # greater than
pset.addPrimitive(operator.le, [pd.Series, pd.Series], bool)   # less than


#  Define the terminals that can be used in the tree
pset.addTerminal(random.randint(5,20), int) # for the window size
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
pop = toolbox.population(n=100)
hof = tools.HallOfFame(10)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)
best_strategy = hof[0]

# How DEAP works:

# In the DEAP framework, individuals in the genetic programming process are represented as trees of functions and terminals. 
# the functions and terminals are created using the PrimitiveTree class, which is a subclass of the gp.PrimitiveTree class.
# the toolbox.expr method is used to create a new individual with a maximum depth of 2 (need to figure out the best depth), 
# by randomly selecting functions and terminals from the primitive set.
# Terminals represent the input values or constants that the functions use as arguments to produce an output.
# (we can have the parameters for the indicators as terminals? the genetic algo can figure out the best parameter and the best way
# to organise the indicators)

# The primitive set is a set of functions and terminals that define the syntax and semantics of the individuals in the genetic programming problem.
# It defines the building blocks of the individuals that will be evolved through the genetic algorithm.
# an individual is a trading rule tree that is represented as a list of functions and terminals
# (its the trading rule, the thing we are trying to optimise)

# the toolbox is a container for the genetic algorithm's components, such as the selection method, crossover and mutation operators, 
# and the evaluation function.
# It allows the user to easily manipulate and organize the various parts of the genetic algorithm. 
# The toolbox is defined using the base.Toolbox class, and it contains methods for adding and removing the various components 
# of the genetic algorithm.

# the creator module is used to create the classes for the fitness and individual.