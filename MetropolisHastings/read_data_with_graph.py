import sys
import os
import re
from os import listdir
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'..')

from MetropolisHastings.parse_tree import ParseTree

cur_dir = os.path.dirname(os.path.realpath(__file__))
file_path = cur_dir + '/' + 'teste/'
print('cur dir = ', file_path)
files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
stats_paths = []
for file in files:
    if 'datafile' in file:
        stats_paths.append(file)

# Sort the files using natural sorting
# Source: https://stackoverflow.com/questions/5967500
#            /how-to-correctly-sort-a-string-with-a-number-inside

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

stats_paths.sort(key=natural_keys)

print('stats_paths')
print(stats_paths)

data = []

for file in stats_paths:

	with open(cur_dir + '/' + 'teste/' + file, "rb") as f:
	    data.append(pickle.load(f))

best_glenn = 0
index_glenn = 0
best_uct = 0
index_uct = 0

victories = []
losses = []
draws = []

victories_against_glenn = []
losses_against_glenn = []
draws_against_glenn = []

# For analysis - Games against UCT
victories_against_UCT = []
losses_against_UCT = []
draws_against_UCT = []

for i in range(len(data)):
	if data[i][3] > best_glenn:
		best_glenn = data[i][3]
		index_glenn = i
	if data[i][6] > best_uct:
		best_uct = data[i][6]
		index_uct = i
	victories.append(data[i][0])
	losses.append(data[i][1])
	draws.append(data[i][2])
	victories_against_glenn.append(data[i][3])
	losses_against_glenn.append(data[i][4])
	draws_against_glenn.append(data[i][5])
	victories_against_UCT.append(data[i][6])
	losses_against_UCT.append(data[i][7])
	draws_against_UCT.append(data[i][8])
print('best_glenn = ', best_glenn)
print('index_glenn = ', index_glenn)
print('best_uct = ', best_uct)
print('index_uct = ', index_uct)
print('filename glenn = ', stats_paths[index_glenn])
print('filename uct = ', stats_paths[index_uct])


x = list(range(len(victories)))

plt.plot(x, victories, color='green', label='Victory')
plt.plot(x, losses, color='red', label='Loss')
plt.plot(x, draws, color='gray', label='Draw')
plt.legend(loc="best")
plt.title("Selfplay generated script against previous script")
plt.xlabel('Iterations')
plt.ylabel('Number of games')
plt.savefig('vs_previous_script.png')

plt.close()


x = list(range(len(victories_against_glenn)))

plt.plot(x, victories_against_glenn, color='green', label='Victory')
plt.plot(x, losses_against_glenn, color='red', label='Loss')
plt.plot(x, draws_against_glenn, color='gray', label='Draw')
plt.legend(loc="best")
plt.title("Hybrid Simulated Annealing - Games against Glenn")
plt.xlabel('Iterations')
plt.ylabel('Number of games')
plt.savefig('vs_glenn.png')

plt.close()

x = list(range(len(victories_against_UCT)))

plt.plot(x, victories_against_UCT, color='green', label='Victory')
plt.plot(x, losses_against_UCT, color='red', label='Loss')
plt.plot(x, draws_against_UCT, color='gray', label='Draw')
plt.legend(loc="best")
plt.title("Hybrid Simulated Annealing - Games against UCT - " + str(50) + " playouts")
plt.xlabel('Iterations')
plt.ylabel('Number of games')
plt.savefig('vs_UCT.png')