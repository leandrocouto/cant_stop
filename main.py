from collections import defaultdict
import random
import time
import numpy as np
import copy
import random
import collections
from game import Board, Game
from utils import valid_positions_channel, finished_columns_channels
from utils import player_won_column_channels, player_turn_channel
from utils import transform_dist_prob, transform_to_input
from utils import transform_actions_to_dist, transform_dataset_to_input
from config import GameConfig, AlphaZeroConfig, NetworkConfig
from statistics import Statistic
from network_uct import Network_UCT
from vanilla_uct import Vanilla_UCT
from network_uct_with_playout import Network_UCT_With_Playout
import tensorflow as tf
from collections import Counter
import sys
import tkinter.filedialog

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import regularizers
from keras.losses import categorical_crossentropy

def define_model(network_config):
    """Neural Network model implementation using Keras + Tensorflow."""
    state_channels = Input(shape = (5,5,6), name='States_Channels_Input')
    valid_actions_dist = Input(shape = (32,), name='Valid_Actions_Input')

    conv = Conv2D(filters=10, kernel_size=2, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(network_config.reg), activation='relu', name='Conv_Layer')(state_channels)
    if network_config.conv_number == 2:
        conv2 = Conv2D(filters=10, kernel_size=2, kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(network_config.reg), activation='relu', name='Conv_Layer2')(conv)
    if network_config.conv_number == 1:
        flat = Flatten(name='Flatten_Layer')(conv)
    else:
        flat = Flatten(name='Flatten_Layer')(conv2)

    # Merge of the flattened channels (after pooling) and the valid action
    # distribution. Used only as input in the probability distribution head.
    merge = concatenate([flat, valid_actions_dist])

    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(100, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(network_config.reg), activation='relu', name='FC_Prob_1')(merge)
    hidden_fc_prob_dist_2 = Dense(100, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(network_config.reg), activation='relu', name='FC_Prob_2')(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(32, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(network_config.reg), activation='softmax', name='Output_Dist')(hidden_fc_prob_dist_2)
    
    #Value of a state
    hidden_fc_value_1 = Dense(100, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(network_config.reg), activation='relu', name='FC_Value_1')(flat)
    hidden_fc_value_2 = Dense(100, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(network_config.reg), activation='relu', name='FC_Value_2')(hidden_fc_value_1)
    output_value = Dense(1, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(network_config.reg), activation='tanh', name='Output_Value')(hidden_fc_value_2)

    model = Model(inputs=[state_channels, valid_actions_dist], outputs=[output_prob_dist, output_value])

    model.compile(loss=['categorical_crossentropy','mean_squared_error'], 
                        optimizer='adam', metrics={'Output_Dist':'categorical_crossentropy', 'Output_Value':'mean_squared_error'})
    return model  

def play_single_game(game_config, alphazero_config, model, second_model, type_of_game):
    """
    - model is the network for the first player (if applicable).
    - second_model is the network for the second player (if applicable).
    - type_of_game is an integer representing who are the players in this
      particular game:
      - type_of_game == 0: if the game to be played is between the same network
                           in the training phase.
      - type_of_game == 1: if the game to be played is between the same network
                           in the training phase using playout simulations.
      - type_of_game == 2: if the game to be played is between the new and 
                           old network.
      - type_of_game == 3: if the game to be played is between the current 
                           network and vanilla UCT.
      - type_of_game == 4: if the game to be played is between two vanilla 
                           UCT instances.
      - type_of_game == 5: if the game to be played is between the new and 
                           old network using playout simulations (instead of
                           fetching the value from the net).
      - type_of_game == 6: if the game to be played is between the network
                           using playout simulations (instead of fetching 
                           the value from the net) and vanilla UCT.
    """
    data_of_a_game = []
    game = Game(game_config)
    is_over = False
    # UCT using new model (network training)
    uct_0 = Network_UCT(model, game_config, alphazero_config)
    # UCT using new model with playout simulations (network training)
    uct_1 = Network_UCT_With_Playout(model, game_config, alphazero_config)
    # UCT using old model (network evaluation)
    uct_2 = Network_UCT(second_model, game_config, alphazero_config)
    # UCT using old model with playout simulations (network evaluation)
    uct_3 = Network_UCT_With_Playout(second_model, game_config, alphazero_config)
    # Vanilla UCT (no networks, testing purpose)
    uct_4 = Vanilla_UCT(alphazero_config)
    infinite_loop = 0
    while not is_over:
        infinite_loop += 1
        # Collecting data for later input to the NN
        channel_valid = valid_positions_channel(game_config)
        channel_finished_1, channel_finished_2 = finished_columns_channels(game, channel_valid)
        channel_won_column_1, channel_won_column_2 = player_won_column_channels(game, channel_valid)
        channel_turn = player_turn_channel(game, channel_valid)
        list_of_channels = [channel_valid, channel_finished_1, channel_finished_2,
                            channel_won_column_1, channel_won_column_2, channel_turn]
        moves = game.available_moves()
        if game.is_player_busted(moves):
            continue
        else:
            if type_of_game == 0:
                chosen_play, dist_probability = uct_0.run_UCT(game)
            elif type_of_game == 1:
                chosen_play, dist_probability = uct_1.run_UCT(game)
            elif type_of_game == 2:
                if game.player_turn == 1:
                    chosen_play, dist_probability = uct_0.run_UCT(game)
                else:
                    chosen_play, dist_probability = uct_2.run_UCT(game)
            elif type_of_game == 3:
                if game.player_turn == 1:
                    chosen_play, dist_probability = uct_0.run_UCT(game)
                else:
                    chosen_play, dist_probability = uct_4.run_UCT(game)
            elif type_of_game == 4:
                chosen_play, dist_probability = uct_4.run_UCT(game)
            elif type_of_game == 5:
                if game.player_turn == 1:
                    chosen_play, dist_probability = uct_1.run_UCT(game)
                else:
                    chosen_play, dist_probability = uct_3.run_UCT(game)
            elif type_of_game == 6:
                if game.player_turn == 1:
                    chosen_play, dist_probability = uct_1.run_UCT(game)
                else:
                    chosen_play, dist_probability = uct_4.run_UCT(game)
            # Collecting data for later input to the NN
            current_play = [list_of_channels, dist_probability]
            data_of_a_game.append(current_play)
            game.play(chosen_play)
        if infinite_loop > alphazero_config.max_game_length:
            network_input = transform_to_input(game, game_config)
            valid_actions_dist = transform_actions_to_dist(game.available_moves())
            _, network_value_output = model.predict([network_input,valid_actions_dist], batch_size=1)
            who_won = 1 if network_value_output[0][0] < 0 else 2
            is_over = True
        else:
            who_won, is_over = game.is_finished()
    return data_of_a_game, who_won

def main():
    # Command line parameters: n_simulations, n_games, alpha_zero, conv_number, use_UCT_playout
    # If the user does not pass any extra command line arguments,
    # then it will open the dialog to generate graphs.
    if len(sys.argv) == 1:
        root = tkinter.Tk()
        root.withdraw()
        file_path = tkinter.filedialog.askopenfilename()
        print(file_path)
        stats = Statistic()
        stats.load_from_file(file_path)
        stats.generate_graphs()
        exit()

    # Cluster configurations
    if int(sys.argv[1]) == 0: n_simulations = 50
    if int(sys.argv[1]) == 1: n_simulations = 100
    if int(sys.argv[1]) == 2: n_simulations = 300
    if int(sys.argv[2]) == 0: n_games = 1
    if int(sys.argv[2]) == 1: n_games = 500
    if int(sys.argv[2]) == 2: n_games = 1000
    if int(sys.argv[3]) == 0: alphazero_iterations = 10
    if int(sys.argv[3]) == 1: alphazero_iterations = 100
    if int(sys.argv[4]) == 0: conv_number = 1
    if int(sys.argv[4]) == 1: conv_number = 2
    if int(sys.argv[5]) == 0: use_UCT_playout = True
    if int(sys.argv[5]) == 1: use_UCT_playout = False

    game_config = GameConfig(n_players = 2, dice_number = 4, dice_value = 3, column_range = [2,6],
                    offset = 2, initial_height = 1)

    alphazero_config = AlphaZeroConfig(c = 10, n_simulations = n_simulations, n_games = n_games, 
                                        n_games_evaluate = 1, max_game_length = 50,
                                        victory_rate = 55, alphazero_iterations = alphazero_iterations, 
                                        use_UCT_playout = use_UCT_playout)

    network_config = NetworkConfig(reg = 0.01, epochs = 1, conv_number = conv_number)

    print('The arguments are: ' , str(sys.argv))
    print('n_sim:', alphazero_config.n_simulations, 'n_gam:', alphazero_config.n_games, 
        'AZ ite:', alphazero_config.alphazero_iterations,
         'conv:', network_config.conv_number, 'use_UCT_playout: ', alphazero_config.use_UCT_playout)

    #Neural network specification
    model = define_model(network_config)
    old_model = define_model(network_config)
    old_model.set_weights(model.get_weights())

    # Stores data from net vs net in evaluation for later analysis.
    data_net_vs_net = []
    # Stores data from net vs uct in evaluation for later analysis.
    data_net_vs_uct = []
    #
    #
    # Main loop of the algorithm
    #
    #
    
    for count in range(alphazero_config.alphazero_iterations):
        print()
        print('ALPHAZERO ITERATION ', count)
        print()
        victory_1 = 0
        victory_2 = 0
        start = time.time()

        # dataset_for_network saves a list of info used as input for the network.
        # A list of: states of the game, distribution probability of the state
        # returned by the UCT and who won the game this state is in.
        dataset_for_network = []

        #
        #
        # Self-play
        #
        #

        for i in range(alphazero_config.n_games):
            if alphazero_config.use_UCT_playout:
                data_of_a_game, who_won = play_single_game(game_config, alphazero_config, 
                                            model, second_model = None, type_of_game = 1)
            else:
                data_of_a_game, who_won = play_single_game(game_config, alphazero_config, model, 
                                            second_model = None, type_of_game = 0)
            print('Self-play - GAME', i ,'OVER - PLAYER', who_won, 'WON')
            if who_won == 1:
                victory_1 += 1
            else:
                victory_2 += 1
            # After the game is finished, we now know who won the game.
            # Therefore, save this info in all instances saved so far
            # for later use as input for the NN.
            for single_game in data_of_a_game:
                single_game.append(who_won)
            dataset_for_network.append(data_of_a_game)
        print('Self-play - Player 1 won', victory_1,'time(s).')
        print('Self-play - Player 2 won', victory_2,'time(s).')

        #
        #
        # Training
        #
        #

        print()
        print('TRAINING LOOP')
        print()
        x_train, y_train = transform_dataset_to_input(dataset_for_network)

        history_callback = model.fit(x_train, y_train, epochs=network_config.epochs, shuffle=True)

        # Saving data
        loss_history = sum(history_callback.history["loss"]) / len(history_callback.history["loss"])
        output_dist_loss_history = sum(history_callback.history["Output_Dist_loss"]) / \
                                    len(history_callback.history["Output_Dist_loss"])
        output_value_loss_history = sum(history_callback.history["Output_Value_loss"]) / \
                                    len(history_callback.history["Output_Value_loss"])
        dist_metric_history = sum(history_callback.history["Output_Dist_categorical_crossentropy"]) / \
                                len(history_callback.history["Output_Dist_categorical_crossentropy"])
        value_metric_history = sum(history_callback.history["Output_Value_mean_squared_error"]) /  \
                                len(history_callback.history["Output_Value_mean_squared_error"])

        #
        #    
        # Model evaluation
        #
        #
        
        print()
        print('MODEL EVALUATION - Network vs. Old Network')
        print()

        # The current network faces the previous one.
        # If it does not win alphazero_config.victory_rate % it completely
        # discards the current network.

        victory_1_eval_net = 0
        victory_2_eval_net = 0

        for i in range(alphazero_config.n_games_evaluate):
            if alphazero_config.use_UCT_playout:
                _, who_won = play_single_game(game_config, alphazero_config, model, 
                                                second_model = old_model, type_of_game = 5)
            else:
                _, who_won = play_single_game(game_config, alphazero_config, model, 
                                            second_model = old_model, type_of_game = 2)
            if who_won == 1:
                victory_1_eval_net += 1
            else:
                victory_2_eval_net += 1

        data_net_vs_net.append((victory_1_eval_net, victory_2_eval_net))

        necessary_won_games = (alphazero_config.victory_rate * alphazero_config.n_games_evaluate) / 100

        if victory_1_eval_net <= necessary_won_games:
            print('New model is worse and won ', victory_1_eval_net)
        else:
            # Overwrites the old model with the current one.
            old_model.set_weights(model.get_weights())
            print('New model is better and won ', victory_1_eval_net)

            # The new model is better. Therefore, evaluate it against
            # vanilla UCT and store the data for later analysis.

            dataset_for_eval = []
            # Saving data
            loss_eval = []
            dist_loss_eval = []
            value_loss_eval = []
            dist_metric_eval = []
            value_metric_eval = []
            victory_1_eval = 0
            victory_2_eval = 0

            print()
            print('MODEL EVALUATION - Network vs. UCT')
            print()

            for i in range(alphazero_config.n_games_evaluate):
                if alphazero_config.use_UCT_playout:
                    data_of_a_game_eval, who_won = play_single_game(game_config, alphazero_config, model, 
                                                            second_model = None, type_of_game = 6)
                else:
                    data_of_a_game_eval, who_won = play_single_game(game_config, alphazero_config, model, 
                                                            second_model = None, type_of_game = 3)
                print('Net vs UCT - GAME', i ,'OVER - PLAYER', who_won, 'WON')
                if who_won == 1:
                    victory_1_eval += 1
                else:
                    victory_2_eval += 1
                # After the game is finished, we now know who won the game.
                # Therefore, save this info in all instances saved so far
                # for later use as input for the NN.
                for single_game in data_of_a_game_eval:
                    single_game.append(who_won)
                dataset_for_eval.append(data_of_a_game_eval)
                print('Net vs UCT - Network won', victory_1_eval,'time(s).')
                print('Net vs UCT - UCT won', victory_2_eval,'time(s).')

                x_train_eval, y_train_eval = transform_dataset_to_input(dataset_for_eval)

                results = model.evaluate(x_train_eval, y_train_eval)
                # Saving data
                loss_eval.append(results[0])
                dist_loss_eval.append(results[1])
                value_loss_eval.append(results[2])
                dist_metric_eval.append(results[3])
                value_metric_eval.append(results[4])
                dataset_for_eval = []

            loss_eval = sum(loss_eval) / len(loss_eval)
            dist_loss_eval = sum(dist_loss_eval) / len(dist_loss_eval)
            value_loss_eval = sum(value_loss_eval) / len(value_loss_eval)
            dist_metric_eval = sum(dist_metric_eval) / len(dist_metric_eval)
            value_metric_eval = sum(value_metric_eval) / len(value_metric_eval)

            

            # Saving data
            data_net_vs_uct.append((loss_history, output_dist_loss_history, output_value_loss_history, 
                dist_metric_history, value_metric_history, victory_1, victory_2, loss_eval, dist_loss_eval, 
                value_loss_eval, dist_metric_eval, value_metric_eval, victory_1_eval, victory_2_eval))

            elapsed_time = time.time() - start
            print('Time elapsed of AlphaZero iteration', count, ': ', elapsed_time)

            stats = Statistic(data_net_vs_net, data_net_vs_uct, alphazero_config, network_config)
            stats.save_to_file(count)
            stats.save_model_to_file(model, count)

if __name__ == "__main__":
    main()