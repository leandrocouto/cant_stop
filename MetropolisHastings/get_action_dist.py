import sys
sys.path.insert(0,'..')
import math
import copy
from game import Game
from play_game_template import play_single_game
from players.vanilla_uct_player import Vanilla_UCT
from players.uct_player import UCTPlayer
from players.random_player import RandomPlayer
from players.rule_of_28 import Rule_of_28_Player
from MetropolisHastings.parse_tree import ParseTree, Node
from MetropolisHastings.DSL import DSL
from MetropolisHastings.metropolis_hastings import MetropolisHastings
from play_game_template import simplified_play_single_game
from Script import Script
import time
import pickle
import os.path
from random import sample
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict

from pylatex import Document, Section, Figure, NoEscape
from pylatex.utils import bold
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def calculate(new_data):
    # Save statistics if the action was chosen
    action_chosen_distribution = get_action_distribution()
    # Save statistics if the action was available but was not chosen
    action_available_not_chosen_distribution = get_action_distribution()
    # Save statistics if the action was available
    action_available_distribution = get_action_distribution()

    for i in range(len(new_data)):
        game = new_data[i][0]
        available_actions = game.available_moves()
        chosen_play = new_data[i][1]
        action_chosen_distribution[chosen_play] += 1
        for action in available_actions:
            if action != chosen_play:
                action_available_not_chosen_distribution[action] += 1
            action_available_distribution[action] += 1
    '''
    print(action_chosen_distribution)
    print()
    print(action_available_not_chosen_distribution)
    print()
    print(action_available_distribution)
    '''
    return action_chosen_distribution, action_available_not_chosen_distribution, action_available_distribution

def get_action_distribution():
    
    action_distribution = OrderedDict()

    for i in range(2, 13):
        for j in range(i, 13):
            action_distribution[(i, j)] = 0

    for i in range(2, 13):
        action_distribution[(i,)] = 0

    action_distribution['y'] = 0
    action_distribution['n'] = 0

    return action_distribution

def sample_data_from_importance_threshold(data, threshold):

    new_data = [d for d in data if d[3] >= threshold]
    return new_data


data = []
# Read the dataset
with open('fulldata_sorted', "rb") as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break
thresholds = [0, 0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75]

for threshold in thresholds:
    initial_data = sample_data_from_importance_threshold(data, threshold)
    action_chosen_distribution, action_available_not_chosen_distribution, action_available_distribution = calculate(initial_data)


    plt.rcdefaults()
    fig, ax = plt.subplots()

    actions = list(action_chosen_distribution.keys())
    #del keys[-2:]
    chosen = list(action_chosen_distribution.values())
    total = list(action_available_distribution.values())
    freq = [0 if total[i] == 0 else chosen[i] / total[i] for i in range(len(chosen))]
    #freq = [chosen[i] / total[i] for i in range(len(chosen))]
    #del values[-2:]
    y_pos = np.arange(len(actions))

    ax.barh(y_pos, freq)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(actions)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Frequency of action being chosen')
    ax.set_title('Action frequency for threshold = ' + str(threshold))
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10.5, 18.5)
    fig.savefig('action_frequency_'+ str(threshold) + '.png', dpi=100)
    #plt.show()