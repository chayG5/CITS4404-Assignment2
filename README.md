# CITS4404-Assignment2
Creating and optimising a trading bot for Bitcoin by evolving the buy and sell functions using strongly typed Genetic Programming. 

# How to Run the Code
```
pip3 install -r requirements.txt
python3 bot.py
```
*Note: Takes around 2 to 3 minutes for running a population of 300 individuals for 50 generations.
You can change the population and generation size in bot.py file. 


***Note that there is a 70% chance for a profit using the preset parameters. The current parameters allow for the program to be executed quickly. However, larger populations and generatiions give better performance. 
# Explanation
1. Get OHLCV data from Kraken Exchange using CCXT package
2. Create a fitness function - the evalaution function
    - the parameters are the buy and sell functions  (evolved by GP - will explain later) and the data (either the training set or the test set)
    - The function produces a fitness value for the buy and sell function 
        - essetially determines how good that combination of buy and sell function is
        - a combination of a buy and sell function can also be called the bot 
        - multiple bots are generated (treated as individuals) which are evolved to create the best bot
    - the parameters for the buy and sell functions is the index number which will be used to get data for a specific time
    - the functions produce a boolean output: where true refers to a buy/sell trigger
    - when the trigger changes from false to true then buy/sell 
    - each time you buy or sell, -2% of current holdings 
    - more detail in the task sheet (pg.6 under Strategy evaluation)
    - a bot is evaluated based on the profit it makes (the fitness score)
3. Intialise the primitive set and the toolbox (this is part of DEAP framework and will be explained under "How DEAP Works)
    - the primitve set contains the functions and arguments that will make up the buy and sell functions
    - the toolbox contains the functions that will be used for the generation of individuals, mutation and crossover (deap_initialiastion file)
4. Split the data retrieved in step 1 into training and test set
    - the training set is used to evolve a bot with the best buy and sell functions
    - the evolved bot is then used on the test set to see how well the it works on unseen data
    - Genetic programming is use to evovle the bot, which is explained in the next step
5. How GP is implemented:
    - Step 1: initialise a population of individuals: in our case the individuals are the buy and sell functions
        - there are two sets of populations: the buy functions and the sell functions
        - the individuals are generated with the help of the primitive set (explained later)
    - Step 2: Assign a fitness value (this uses our evaluate function) to all the individuals generated
        - a set of individuals (a combination of buy and sell function) is sent to the evaluation function
        - the evaluation function assigns a value to that combination 
    - Step 3: Clone the parents with higher fitness so offsprings can be created
        - crossover and mutation is applied to the cloned individuals to create offsprings
        - crossover and mutation introduce variety into the popuation so that more combinations can be tested
        - The offsprings are set as the new parents
        - Step 2 and Step 3 are repeated for n generations
6. After all the generations are run, the best individuals (buy and sell function) are picked
    - The buy and sell function combined together will create the trading bot which will be used on the test set to determine how well the bot works on unseen data. 

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

