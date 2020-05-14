from statistics import Statistic
from players.vanilla_uct_player import Vanilla_UCT
from players.net_uct_player import Network_UCT
from players.net_uct_with_playout_player import Network_UCT_With_Playout
from players.random_player import RandomPlayer
from experiment import Experiment
import sys
import os
from os import listdir
from os.path import isfile, join
import re
import pickle
import gc
from models import define_model, define_model_experimental
from keras.models import load_model
import multiprocessing

def get_last_iteration(folder):
    """ Return which iteration AZ should start from. """
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = cur_dir + '/' + folder
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    stats_paths = []
    for file in files:
        if 'h5' not in file and '_player' not in file:
            stats_paths.append(file)

    # Sort the files using natural sorting
    # Source: https://stackoverflow.com/questions/5967500
    #            /how-to-correctly-sort-a-string-with-a-number-inside

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    stats_paths.sort(key=natural_keys)

    last_iteration = int(stats_paths[-1].rsplit('_', 1)[1])
    return last_iteration + 1

def get_last_player(folder):
    """ Return the last player from the last AZ iteration. """

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = cur_dir + '/' + folder
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    networks_paths = []
    players_paths = []
    for file in files:
        if 'h5' in file:
            networks_paths.append(file)
        elif '_player' in file:
            players_paths.append(file)

    # Sort the files using natural sorting
    # Source: https://stackoverflow.com/questions/5967500
    #            /how-to-correctly-sort-a-string-with-a-number-inside

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    networks_paths.sort(key=natural_keys)
    players_paths.sort(key=natural_keys)

    last_model = file_path + '/' + networks_paths[-1]
    last_model = load_model(last_model)
    player = file_path + '/' + players_paths[-1]
    with open(player, 'rb') as file:
        player = pickle.load(file)
    player.network = last_model
    return player

def main():
    # Command line parameters: n_simulations, n_games, alpha_zero, 
    # conv_number, use_UCT_playout

    # Cluster configurations
    if int(sys.argv[1]) == 0: n_simulations = 10
    if int(sys.argv[1]) == 1: n_simulations = 100
    if int(sys.argv[1]) == 2: n_simulations = 250
    if int(sys.argv[1]) == 3: n_simulations = 500
    if int(sys.argv[2]) == 0: n_games = 10
    if int(sys.argv[2]) == 1: n_games = 100
    if int(sys.argv[2]) == 2: n_games = 250
    if int(sys.argv[2]) == 3: n_games = 500
    if int(sys.argv[3]) == 0: alphazero_iterations = 20
    if int(sys.argv[4]) == 0: conv_number = 1
    if int(sys.argv[4]) == 1: conv_number = 2
    if int(sys.argv[5]) == 0: use_UCT_playout = True
    if int(sys.argv[5]) == 1: use_UCT_playout = False

    #Config parameters

    c = 1
    epochs = 1
    reg = 0.01
    n_games_evaluate = 100
    victory_rate = 55
    mini_batch = 2048
    n_training_loop = 1000
    dataset_size = 1000
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
    
    

    

    
    folder = str(n_simulations) + '_' + str(n_games) \
                + '_' + str(alphazero_iterations) + '_' + str(conv_number) \
                + '_' + str(use_UCT_playout)
    file_name = folder + '.txt'

    with open(file_name, 'a') as f:
        print('The arguments are: ' , str(sys.argv), file=f)

    begin_from = 0
    # This means there are already data available, therefore continues from 
    # where it first left off.
    if os.path.isdir('training_data_' + folder):
        begin_from = get_last_iteration('training_data_' + folder)
        # Main loop
        for count in range(begin_from, alphazero_iterations):
            # Useful only for the first iteration since there will be no data
            # to read the player info from.
            player = get_last_player('training_data_' + folder)
            experiment = Experiment(
                        n_players = n_players, 
                        dice_number = dice_number, 
                        dice_value = dice_value, 
                        column_range = column_range, 
                        offset = offset, 
                        initial_height = initial_height, 
                        max_game_length = max_game_length,
                        n_cores = n_cores
                        )

            experiment.play_alphazero(
                                player,
                                use_UCT_playout = use_UCT_playout, 
                                reg = reg,
                                epochs = epochs, 
                                conv_number = conv_number,
                                alphazero_iterations = alphazero_iterations,
                                mini_batch = mini_batch,
                                n_training_loop = n_training_loop, 
                                n_games = n_games, 
                                n_games_evaluate = n_games_evaluate, 
                                victory_rate = victory_rate,
                                dataset_size = dataset_size,
                                iteration = count
                                )

    # First time running the algorithm, there's no data available
    else:
        #Neural network specification
        current_model = define_model(
                                reg = reg, 
                                conv_number = conv_number, 
                                column_range = column_range, 
                                offset = offset, 
                                initial_height = initial_height, 
                                dice_value = dice_value
                                )
        if use_UCT_playout:
            player = Network_UCT_With_Playout(
                                    c = c, 
                                    n_simulations = n_simulations,  
                                    column_range = column_range, 
                                    offset = offset, 
                                    initial_height = initial_height, 
                                    dice_value = dice_value,
                                    network = current_model
                                    )
        else:
            player = Network_UCT(
                                c = c, 
                                n_simulations = n_simulations, 
                                column_range = column_range,
                                offset = offset, 
                                initial_height = initial_height, 
                                dice_value = dice_value, 
                                network = current_model
                                )
        # Main loop
        for count in range(alphazero_iterations):
            if os.path.isdir('training_data_' + folder):
                player = get_last_player('training_data_' + folder)
            experiment = Experiment(
                        n_players = n_players, 
                        dice_number = dice_number, 
                        dice_value = dice_value, 
                        column_range = column_range, 
                        offset = offset, 
                        initial_height = initial_height, 
                        max_game_length = max_game_length,
                        n_cores = n_cores
                        )
            experiment.play_alphazero(
                                player,  
                                use_UCT_playout = use_UCT_playout, 
                                reg = reg,
                                epochs = epochs, 
                                conv_number = conv_number,
                                alphazero_iterations = alphazero_iterations,
                                mini_batch = mini_batch,
                                n_training_loop = n_training_loop, 
                                n_games = n_games, 
                                n_games_evaluate = n_games_evaluate, 
                                victory_rate = victory_rate,
                                dataset_size = dataset_size,
                                iteration = count
                                )
            del experiment
            del player
            gc.collect()

if __name__ == "__main__":
    main()