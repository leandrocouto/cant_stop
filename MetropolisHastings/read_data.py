import sys
import os
import re
from os import listdir
from os.path import isfile, join
import pickle

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

for i in range(len(data)):
	if data[i][8] > best_glenn:
		best_glenn = data[i][8]
		index_glenn = i
	if data[i][11] > best_uct:
		best_uct = data[i][11]
		index_uct = i
print('best_glenn = ', best_glenn)
print('index_glenn = ', index_glenn)
print('best_uct = ', best_uct)
print('index_uct = ', index_uct)
print('filename glenn = ', stats_paths[index_glenn])
print('filename uct = ', stats_paths[index_uct])