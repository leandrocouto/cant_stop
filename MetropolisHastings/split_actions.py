import pickle
import sys
sys.path.insert(0,'..')
import math
import copy
from game import Game

data = []
dataset_name = 'fulldata_sorted'

# Read the dataset
with open(dataset_name, "rb") as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break

string_data = []
column_data = []

for d in data:
    if d[0].available_moves()[0] in ['y', 'n']:
        string_data.append(d)
    else:
        column_data.append(d)

with open('fulldata_sorted_string', "ab") as f:
    for d in string_data:
        pickle.dump(d, f)

with open('fulldata_sorted_column', "ab") as f:
    for d in column_data:
        pickle.dump(d, f)