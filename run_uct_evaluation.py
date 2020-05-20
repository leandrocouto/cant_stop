from statistics import Statistic
from players.vanilla_uct_player import Vanilla_UCT
from players.net_uct_player import Network_UCT
from players.net_uct_with_playout_player import Network_UCT_With_Playout
from players.random_player import RandomPlayer
from experiment import Experiment
from statistics import Statistic
import sys
import pickle
import os
from os import listdir
from os.path import isfile, join
import re
import multiprocessing

def get_last_iteration(folder):
    """ Return the next iteration of UCT evaluation to be made. """

    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    stats_paths = []
    for file in files:
        if '_data' not in file and '.txt' not in file:
            stats_paths.append(file)
    return len(stats_paths)

def get_list_of_networks(n_simulations, n_games, alphazero_iterations,
    conv_number, use_UCT_playout):
    """
    Return the AlphaZeroPlayer and Statitisc of the AZ iteration that won 
    against the previous network.
    Parameters passed are used only to locate the source folder of the 
    experiment.
    """
    
    source_folder = 'training_data_' + str(n_simulations) + '_' + \
                    str(n_games) + '_' + str(alphazero_iterations) + '_' +\
                    str(conv_number) + '_' + str(use_UCT_playout) + \
                    '/won' 
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # Directory where new network was better than the old one
    file_path = cur_dir + '/' + source_folder
    
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    networks_paths = []
    players_paths = []
    stats_paths = []
    for file in files:
        if 'modelweights' in file:
            networks_paths.append(file_path + '/' + file)
        elif '_player' in file:
            players_paths.append(file_path + '/' + file)
        else:
            stats_paths.append(file_path + '/' + file)

    # Sort the files using natural sorting
    # Source: https://stackoverflow.com/questions/5967500
    #            /how-to-correctly-sort-a-string-with-a-number-inside

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    networks_paths.sort(key=natural_keys)
    players_paths.sort(key=natural_keys)
    stats_paths.sort(key=natural_keys)

    network_weights = []
    # Save the weights
    for net_path in networks_paths:
        with open(net_path, 'rb') as file:
            weight = pickle.load(file)
            network_weights.append(weight)

    # Instantiate players
    players = []
    for player_path in players_paths:
        with open(player_path, 'rb') as file:
            player = pickle.load(file)
            players.append(player)

    # Instantiate stats
    stats = []
    for stat_path in stats_paths:
        with open(stat_path, 'rb') as file:
            stat = Statistic()
            stat.load_from_file(stat_path)
            stats.append(stat)
    # Get the prefix name of the files (Ex: 2_10_10_1_False_9)
    prefix_names = []
    for p in players_paths:
        file_name = p.rsplit('/', 1)[-1].rsplit('_', 1)[0]
        prefix_names.append(file_name)

    return players, network_weights, stats, prefix_names, file_path

def main():

    # Cluster configurations
    if int(sys.argv[1]) == 0: n_simulations = 10
    if int(sys.argv[1]) == 1: n_simulations = 100
    if int(sys.argv[1]) == 2: n_simulations = 250
    if int(sys.argv[1]) == 3: n_simulations = 500
    if int(sys.argv[2]) == 0: n_games = 50
    if int(sys.argv[2]) == 1: n_games = 100
    if int(sys.argv[2]) == 2: n_games = 250
    if int(sys.argv[2]) == 3: n_games = 500
    if int(sys.argv[3]) == 0: alphazero_iterations = 10
    if int(sys.argv[4]) == 0: conv_number = 1
    if int(sys.argv[4]) == 1: conv_number = 2
    if int(sys.argv[5]) == 0: use_UCT_playout = True
    if int(sys.argv[5]) == 1: use_UCT_playout = False

    #Config parameters
    n_games_evaluate = 10
    reg = 0.01
    n_cores = multiprocessing.cpu_count()

    '''
    # Toy version
    column_range = [2,6]
    offset = 2
    initial_height = 1
    n_players = 2
    dice_number = 4
    dice_value = 3
    max_game_length = 50
    '''
    # Original version
    column_range = [2,12]
    offset = 2
    initial_height = 2 
    n_players = 2
    dice_number = 4
    dice_value = 6 
    max_game_length = 500

    players, weights, stats, prefix_names, net_dir = get_list_of_networks(
                                                        n_simulations,
                                                        n_games,
                                                        alphazero_iterations,
                                                        conv_number,
                                                        use_UCT_playout
                                                        )
    
    # This means there are already data available, therefore continues from 
    # where it first left off.
    if os.path.isdir(net_dir + '/results_uct'):
        begin_from = get_last_iteration(net_dir + '/results_uct')
    # First time running the UCT evaluations, there's no data available
    else:
        begin_from = 0

    for i in range(begin_from, len(players)):
        experiment = Experiment(
                        n_players = n_players, 
                        dice_number = dice_number, 
                        dice_value = dice_value, 
                        column_range = column_range, 
                        offset = offset, 
                        initial_height = initial_height, 
                        max_game_length = max_game_length,
                        reg = reg,
                        conv_number = conv_number,
                        n_cores = n_cores
                        )
        n_simulations = players[i].n_simulations
        uct_eval_1 = Vanilla_UCT(
                                    c = players[i].c, 
                                    n_simulations = round(0.25 * n_simulations)
                                    )
        uct_eval_2 = Vanilla_UCT(
                                    c = players[i].c, 
                                    n_simulations = round(0.5 * n_simulations)
                                    )
        uct_eval_3 = Vanilla_UCT(
                                    c = players[i].c, 
                                    n_simulations =  n_simulations
                                    )
        UCTs_eval = [uct_eval_1, uct_eval_2, uct_eval_3]

        experiment.play_network_versus_UCT(
                                            players[i],
                                            weights[i],
                                            stats[i],
                                            net_dir,
                                            prefix_names[i], 
                                            UCTs_eval,
                                            n_games_evaluate
                                            )

if __name__ == "__main__":
    main()