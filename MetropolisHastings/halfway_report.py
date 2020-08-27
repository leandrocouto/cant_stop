import pickle
import sys
sys.path.insert(0,'..')
from MetropolisHastings.parse_tree import ParseTree


with open('datafile_iteration_0', "rb") as f:
    data = pickle.load(f)

print(data)

print(data[-1].generate_program())