from genetic_program import genetic_program, test_data, evaluate
from deap import tools
from helper import *
import matplotlib.pyplot as plt

# run the genetic program: parameter: population size, number of generations
pop_buy, pop_sell, x_gen, y_avgProfit = genetic_program(300, 50)

# get the best buy and sell functions
best_buy = tools.selBest(pop_buy, k=1)[0]
best_sell = tools.selBest(pop_sell, k=1)[0]

# average profit per generation plot
plt.plot(x_gen, y_avgProfit)

# set the x and y axis labels
plt.xlabel('Generations')
plt.ylabel('Average Profit')

# set the title of the plot
plt.title('The Average Profit Per Generation')
plt.show()

# get the profit using the evolved bot
final = evaluate(best_buy, best_sell, test_data)
print()
print("Buy Function: ", best_buy)
print()
print("Sell Function: ", best_sell)
print()
print("Final Profit: ", final)