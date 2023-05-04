from testing import genetic_program, test_data, evaluate
from deap import tools
from get_data import *
import matplotlib.pyplot as plt

pop_buy, pop_sell, x_gen, y_avgProfit = genetic_program(500, 30)
# get the best buy and sell functions
best_buy = tools.selBest(pop_buy, k=1)[0]
best_sell = tools.selBest(pop_sell, k=1)[0]

# results
plt.plot(x_gen, y_avgProfit)

# set the x and y axis labels
plt.xlabel('Generations')
plt.ylabel('average profit')
# set the title of the plot
plt.title('The average profit per generation')
plt.show()
print(x_gen)
print(y_avgProfit)

final = evaluate(best_buy, best_sell, test_data)

print("Test set: ", final)