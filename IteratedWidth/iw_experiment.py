import sys
import random
import time
import pickle
import numpy as np
import math
import os
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from IteratedWidth.selfplay import SelfPlay
from IteratedWidth.toy_DSL import ToyDSL
from players.random_glenn_player import RandomGlennPlayer
from IteratedWidth.sketch import Sketch
from IteratedWidth.iterated_width import IteratedWidth
from IteratedWidth.breadth_first_search import BFS
from IteratedWidth.simulated_annealing import SimulatedAnnealing
from play_game_template import simplified_play_single_game
from game import Game

if __name__ == "__main__":
    # Search algorithm -> 0 = IW, 1 = BFS
    search_type = int(sys.argv[1])
    # Local Search algorithm -> 0 = random, 1 = random + SA, 2 = MC simulations
    LS_type = int(sys.argv[2])

    n_SA_iterations_options = [100, 500, 1000]
    n_SA_iterations = n_SA_iterations_options[int(sys.argv[3])]

    # k and n_states are paired together in order to make a fair comparison
    # between algorithms. This is because IW if k is low will have a low number
    # of states returned.
    k_options = [4, 5, 6]
    k = k_options[int(sys.argv[4])]
    n_states_options = [500, 2000, 6000]
    n_states = n_states_options[int(sys.argv[4])]

    n_games = 100
    init_temp = 1
    d = 1
    max_game_rounds = 500
    max_nodes = 200
    n_MC_simulations = 100
    inner_SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, False)
    outer_SA = SimulatedAnnealing(10000, max_game_rounds, n_games, init_temp, d, True)
    

    dsl = ToyDSL()
    
    if search_type == 0:
        # IW
        tree = ParseTree(dsl=dsl, max_nodes=max_nodes, k=k, is_IW=True)
        search_algo = IteratedWidth(tree, n_states, k)
        suffix = 'IW' + '_LS' + str(LS_type) + '_SA' + str(n_SA_iterations) + '_ST' + str(n_states) + '_k' + str(k) + '_GA' + str(n_games)
    elif search_type == 1:
        # BFS
        tree = ParseTree(dsl=dsl, max_nodes=max_nodes, k=k, is_IW=False)
        search_algo = BFS(tree, n_states)
        suffix = 'BFS' + '_LS' + str(LS_type) + '_SA' + str(n_SA_iterations) + '_ST' + str(n_states) + '_k' + str(k) + '_GA' + str(n_games)

    SP = SelfPlay(n_games, n_MC_simulations, max_game_rounds, max_nodes, search_algo, inner_SA, outer_SA, dsl, LS_type, suffix)
    start = time.time()
    SP.run()
    elapsed = time.time() - start
    print('Elapsed = ', elapsed)