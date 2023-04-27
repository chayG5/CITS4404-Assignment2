# CITS4404-Assignment2
Trading Bot for Bitcoin by evolving the buy and sell functions using the DEAP framework. 

# How DEAP Works:

In the DEAP framework, individuals in the genetic programming process are represented as trees of functions and terminals. 
the functions and terminals are created using the PrimitiveTree class, which is a subclass of the gp.PrimitiveTree class.
the toolbox.expr method is used to create a new individual with a maximum depth of 2 (need to figure out the best depth), 
by randomly selecting functions and terminals from the primitive set.
Terminals represent the input values or constants that the functions use as arguments to produce an output.
(we can have the parameters for the indicators as terminals? the genetic algo can figure out the best parameter and the best way
to organise the indicators)

The primitive set is a set of functions and terminals that define the syntax and semantics of the individuals in the genetic programming problem.
It defines the building blocks of the individuals that will be evolved through the genetic algorithm.
an individual is a trading rule tree that is represented as a list of functions and terminals
(its the trading rule, the thing we are trying to optimise)

the toolbox is a container for the genetic algorithm's components, such as the selection method, crossover and mutation operators, 
and the evaluation function.
It allows the user to easily manipulate and organize the various parts of the genetic algorithm. 
The toolbox is defined using the base.Toolbox class, and it contains methods for adding and removing the various components 
of the genetic algorithm.

the creator module is used to create the classes for the fitness and individual.