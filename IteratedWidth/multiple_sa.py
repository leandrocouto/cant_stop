import sys
import random
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import os
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from players.random_glenn_player import RandomGlennPlayer
from IteratedWidth.sketch import Sketch
from IteratedWidth.iterated_width import IteratedWidth
from IteratedWidth.breadth_first_search import BFS
from IteratedWidth.simulated_annealing import SimulatedAnnealing
from play_game_template import simplified_play_single_game
from game import Game


best_indexes = [446, 400, 5976, 167, 165, 833]
files = ['closed_list_bfs_k4','closed_list_bfs_k5','closed_list_bfs_k6','closed_list_iw_k4','closed_list_iw_k5','closed_list_iw_k6']
n_games = 100
init_temp = 1
d = 1
max_game_rounds = 500

data = []

for file in files:

    with open(file, "rb") as f:
        data.append(pickle.load(f))

i = int(sys.argv[1])

print('programa antes')
print(data[i][best_indexes[i]].generate_program())
print()
best_player_state = data[i][best_indexes[i]]
best_player_state.finish_tree_randomly()
print('programa depois')
print(best_player_state.generate_program())
print()
best_player = Sketch(best_player_state.generate_program()).get_object()

outer_SA = SimulatedAnnealing(10000, max_game_rounds, n_games, init_temp, d, True)

suffix = files[i]
folder = os.path.dirname(os.path.realpath(__file__)) + '/'+suffix + '_folder/'
if not os.path.exists(folder):
    os.makedirs(folder)
filename = folder + '/log_file_'+ suffix + '.txt'
outer_SA.folder = folder
outer_SA.filename = filename
outer_SA.suffix = suffix
outer_SA.run(best_player_state, best_player)