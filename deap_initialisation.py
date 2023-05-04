from deap import gp, base, creator, tools
from helper import *
import operator

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

# the indicators
pset.addPrimitive(comparemacd, [int], Bool)
pset.addPrimitive(rsi_30, [int], Bool)
pset.addPrimitive(rsi_70, [int], Bool)
pset.addPrimitive(detectbbh, [int], Bool)
pset.addPrimitive(Stoch_20, [int], Bool)
pset.addPrimitive(Stoch_80, [int], Bool)
pset.addPrimitive(detectObv, [int], Bool)
pset.addPrimitive(sma_20_50, [int], Bool)
pset.addPrimitive(sma_50_20, [int], Bool)

# operators 
pset.addPrimitive(operator.and_, [Bool, Bool], Bool)
pset.addPrimitive(operator.or_, [Bool, Bool], Bool)
pset.addPrimitive(operator.not_, [Bool], Bool)

# rename the input argument 
pset.renameArguments(ARG0="time")

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