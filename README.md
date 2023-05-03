# CITS4404-Assignment2
Trading Bot for Bitcoin by evolving the buy and sell functions using Genetic Programming. 

# Explanation
1. Get OHLCV data from Kraken Exchange using CCXT package (get_data.py file)
2. Create a fitness function (the evaluate function in trading_rule.py file)
    - the parameters are the buy and sell functions (evolved by GP - will explain later)
    - The function produces a fitness value for the buy and sell function -> how good the buy/sell function is
    - the parameters for the buy and sell functions are the index number and the TA indicators for the 5 and 10 period 
    - the functions produce a boolean output: where true refers to a buy/sell trigger
    - when the trigger changes from false to true then buy/sell 
    - each time you buy or sell, -2% of current holdings 
    - more detail in the task sheet (pg.6 under Strategy evaluation)
    - the combination of buy and sell function that is received, is evaluated based on the profit it generates. 
3. Intialise the primitive set and the toolbox (this is part of DEAP framework which is explained below)
4. How GP is implemented:
    - Step 1: initialise a population of individuals: in our case the individuals are the buy and sell functions
        - there are two sets of populations: the buy functions and the sell functions
        - the individuals are generated with the help of the primitive set (explained later)
    - Step 2: Assign a fitness value (this uses our evaluate function) to all the individuals generated
        - a set of individuals (a combination of buy and sell function) is sent to the evaluation function
        - the evaluation function assigns a value to that combination 
    - Step 3: Clone the parents so offsprings can be created
        - crossover and mutation is applied to the cloned individuals to create offsprings
        - crossover and mutation allow for more possible buy/sell functions
        - The offsprings are set as the new parents
        - Step 2 and Step 3 are repeated for n generations
5. After all the generations are run, the best individuals (buy and sell function) are picked
    - The buy and sell function combined together is the trading bot (tells when to buy and when to sell)

# How DEAP Works:

- In the DEAP framework, individuals in the genetic programming process are represented as trees of "functions" and "terminals". 
    - in our case, an individual is the buy/sell function
- The "functions" and "terminals" are created using the primitive set
- The primitive set is a set of functions and terminals that define the syntax and semantics of the individuals in the genetic programming problem. 
    - It defines the building blocks of the individuals that will be evolved through the genetic algorithm.
    - Basically, when the population is initialised (mentioned above), the individuals in the population are created using the functions and terminals in the primitve set
    - Example of a function could be a TA indicator, greater than operator, multiply and so on..
    - Terminals represent the input values or constants that the functions use as arguments to produce an output e.g. the window size for TA indicator
- The toolbox is a container for the genetic algorithm's components, such as the selection method, crossover and mutation operators, 
and the evaluation function.
    - The toolbox.expr method is used to create a new individual by randomly selecting functions and terminals from the primitive set.
