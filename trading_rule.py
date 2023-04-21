from get_data import *
from deap import gp, base, creator, tools

#data = add_taIndicators().to_csv("OHLCV_data")

# Define the functions that can be used in the tree
def buy(open_price, close_price):
    # code for when to buy
    return 1

def sell(open_price, close_price):
    # code for when to sell
    return 1

# Define the evaluation function that maps a trading rule tree to a fitness value
def evaluate(individual):
    func = gp.compile(individual, pset)
    # Apply the trading rule

# A class to represent the fitness of an individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# A class to represent an individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initialize the primitive set
pset = gp.PrimitiveSet("MAIN", arity=2)

# Define the functions that can be used in the tree
pset.addPrimitive(buy, arity=2)
pset.addPrimitive(sell, arity=2)

#  Define the terminals that can be used in the tree
pset.addTerminal(add_taIndicators()["rsi"])
pset.addTerminal(add_taIndicators()["bb_high"])
pset.addTerminal(add_taIndicators()["bb_low"])

# Initialize the toolbox
toolbox = base.Toolbox()
# a generator function that generates trees of functions and operands using the primitive set pset.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) # min_ and max_ are the minimum and maximum heights of the generated trees.
# a function that creates a new individual from a generator function (expr).
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# a function that creates a population of individuals from a generator function (individual).
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# a function that compiles an individual into a callable function.
toolbox.register("compile", gp.compile, pset=pset)
# a function that evaluates the fitness of an individual.
toolbox.register("evaluate", evaluate)
# a crossover operator that applies one-point crossover to two individuals.
toolbox.register("mate", gp.cxOnePoint)
# a mutation operator that applies uniform mutation to an individual.
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
# a selection operator that selects individuals using tournament selection.
toolbox.register("select", tools.selTournament, tournsize=3)


# How DEAP works:

# In the DEAP framework, individuals in the genetic programming process are represented as trees of functions and terminals. 
# the functions and terminals are created using the PrimitiveTree class, which is a subclass of the gp.PrimitiveTree class.
# the toolbox.expr method is used to create a new individual with a maximum depth of 2 (need to figure out the best depth), 
# by randomly selecting functions and terminals from the primitive set.

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