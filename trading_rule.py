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

# Initialize the primitive set
pset = gp.PrimitiveSet("MAIN", arity=2)

# Add the functions to the primitive set
pset.addPrimitive(buy, arity=2)
pset.addPrimitive(sell, arity=2)

# Define the terminals that can be used in the tree
pset.addTerminal(add_taIndicators()["rsi"])
pset.addTerminal(add_taIndicators()["bb_high"])
pset.addTerminal(add_taIndicators()["bb_low"])

# Define the evaluation function that maps a trading rule tree to a fitness value
def evaluate(individual):
    func = gp.compile(individual, pset)
    # Apply the trading rule

# Define the fitness function that evaluates the performance of a trading rule tree
def fitness(individual):
    # Calculate the fitness value based on the performance of the trading rule tree
    return 1