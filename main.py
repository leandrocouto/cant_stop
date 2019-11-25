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
from utils import transform_actions_to_dist
from uct import MCTS
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import regularizers
from keras.losses import categorical_crossentropy
from collections import Counter

def custom_loss_crossentropy(y_true, y_pred):
    #y_true = K.print_tensor(y_true, message='y_true = ')
    #y_pred = K.print_tensor(y_pred,message='y_pred = ')
    return K.categorical_crossentropy(y_true, y_pred)
    #return K.binary_crossentropy(y_true, y_pred)


def custom_loss_mse(y_true, y_pred):
    y_true = K.print_tensor(y_true, message='y_true = ')
    y_pred = K.print_tensor(y_pred,message='y_pred = ')
    return K.mean(K.square(y_pred - y_true), axis=-1)

def define_model(config):
    """Neural Network model implementation using Keras + Tensorflow."""
    state_channels = Input(shape = (5,5,6), name='States_Channels_Input')
    valid_actions_dist = Input(shape = (32,), name='Valid_Actions_Input')

    conv = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='Conv_Layer')(state_channels)
    #conv2 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='Conv_Layer2')(conv)
    #conv3 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='Conv_Layer3')(conv2)
    #conv2 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), padding='same', activation='relu', name='Conv_Layer2')(conv)
    #conv3 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), padding='same', activation='relu', name='Conv_Layer3')(conv2)
    #conv4 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), padding='valid', activation='relu', name='Conv_Layer4')(conv3)
    #pool = MaxPooling2D(pool_size=(2, 2), name='Pooling_Layer')(conv)
    flat = Flatten(name='Flatten_Layer')(conv)

    # Merge of the flattened channels (after pooling) and the valid action
    # distribution. Used only as input in the probability distribution head.
    merge = concatenate([flat, valid_actions_dist])

    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='FC_Prob_1')(merge)
    hidden_fc_prob_dist_2 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='FC_Prob_2')(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(32, kernel_regularizer=regularizers.l2(config.reg), activation='softmax', name='Output_Dist')(hidden_fc_prob_dist_2)
    
    #Value of a state
    hidden_fc_value_1 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='FC_Value_1')(flat)
    hidden_fc_value_2 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='FC_Value_2')(hidden_fc_value_1)
    output_value = Dense(1, kernel_regularizer=regularizers.l2(config.reg), activation='tanh', name='Output_Value')(hidden_fc_value_2)

    model = Model(inputs=[state_channels, valid_actions_dist], outputs=[output_prob_dist, output_value])#final_output)

    #model.compile(loss={'Output_Dist': custom_loss_crossentropy, 'Output_Value': custom_loss_mse}, loss_weights={'Output_Dist':0.5,
    #      'Output_Value':0.5}, optimizer='adam', metrics=['accuracy'])
    #model.compile(loss=['kullback_leibler_divergence','mean_squared_error'], 
                        #optimizer='adam', metrics={'Output_Dist':'binary_accuracy', 'Output_Value':'accuracy'})
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], 
                        optimizer='adam', metrics=['binary_accuracy'])#{'Output_Dist':'binary_accuracy', 'Output_Value':'accuracy'})
    return model
    
class Config:
    """ General configuration class for the game board, UCT and NN """
    def __init__(self, c, n_simulations, n_games, maximum_game_length, n_players,
                     dice_number, dice_value, column_range, offset, initial_height,
                    mini_batch, sample_size, n_games_evaluate, victory_rate,
                    alphazero_iterations, reg):
        """
        - c is the constant the balance exploration and exploitation.
        - n_simulations is the number of simulations made in the UCT algorithm.
        - n_games is the number of games played in the self-play scheme.
        - maximum_game_length is the max number of plays in a game by both
          players (avoids infinite loop).
        - n_players is the number of players (At the moment, only 2 is possible).
        - dice_number is the number of dice used in the Can't Stop game.
        - dice_value is the number of sides of a single die.
        - column_range is a list denoting the range of the board game columns.
        - offset is the height difference between columns.
        - initial_height is the height of the columns at the border of the board.
        - mini_batch is the number of inputs selected to train the network.
        - sample_size is the total number of inputs mini_batch is sampled from.
        - n_games_evaluate is the number of the self-played games to evaluate
          the new network vs the old one.
        - victory_rate is the % of victories necessary for the new network to
          overwrite the previously one.
        - alphazero_iterations is the total number of iterations of the learning
          algorithm: selfplay -> training loop -> evaluate network (repeat).
        - reg is the L2 regularization parameter.
        """
        self.c = c
        self.n_simulations = n_simulations
        self.n_games = n_games
        self.maximum_game_length = maximum_game_length
        self.n_players = n_players
        self.dice_number = dice_number
        self.dice_value = dice_value
        self.column_range = column_range 
        self.offset = offset
        self.initial_height = initial_height
        self.mini_batch = mini_batch
        self.sample_size = sample_size
        self.n_games_evaluate = n_games_evaluate
        self.victory_rate = victory_rate
        self.alphazero_iterations = alphazero_iterations
        self.reg = reg

def main():
    victory_1 = 0
    victory_2 = 0
    config = Config(c = 1, n_simulations = 10, n_games = 100, maximum_game_length = 50,
                    n_players = 2, dice_number = 4, dice_value = 3, 
                    column_range = [2,6], offset = 2, initial_height = 1, 
                    mini_batch = 2, sample_size = 100, n_games_evaluate= 50, 
                    victory_rate = .55, alphazero_iterations = 1, reg = 0.01)
    #Neural network specification
    model = define_model(config)
    # summarize layers
    #print(model.summary())
    #
    # Main loop of the algorithm
    #
    for count in range(config.alphazero_iterations):
        print('ALPHAZERO ITERATION ', count)
        dataset_for_network = []
        start = time.time()
        #
        # Self-play
        #
        for i in range(config.n_games):
            data_of_a_game = []
            game = Game(config)
            is_over = False
            uct = MCTS(config, model)
            print('Game', i, 'has started.')
            infinite_loop = 0
            while not is_over:
                infinite_loop += 1
                # Collecting data for later input to the NN
                channel_valid = valid_positions_channel(config)
                channel_finished_1, channel_finished_2 = finished_columns_channels(game, channel_valid)
                channel_won_column_1, channel_won_column_2 = player_won_column_channels(game, channel_valid)
                channel_turn = player_turn_channel(game, channel_valid)
                list_of_channels = [channel_valid, channel_finished_1, channel_finished_2,
                                    channel_won_column_1, channel_won_column_2, channel_turn]
                moves = game.available_moves()
                if game.is_player_busted(moves):
                    continue
                else:
                    #print('antes run')
                    if game.player_turn == 1:
                        chosen_play, dist_probability = uct.run_mcts(game)
                    else:
                        chosen_play, dist_probability = uct.run_mcts(game)
                    #print('dps run')
                    # Collecting data for later input to the NN
                    current_play = [list_of_channels, dist_probability]
                    data_of_a_game.append(current_play)
                    #print('antes play, chosen play = ', chosen_play, 'avaiable plays: ', moves)
                    #print('player ', game.player_turn)
                    #print('finished columns: ', game.finished_columns)
                    #print('won columns: ', game.player_won_column)
                    game.play(chosen_play)
                    #game.board_game.print_board([])
                    #print('dps play')
                if infinite_loop > config.maximum_game_length:
                    #print('chegou no maximo')
                    network_input = transform_to_input(game, config)
                    valid_actions_dist = transform_actions_to_dist(game.available_moves())
                    _, network_value_output = model.predict([network_input,valid_actions_dist], batch_size=1)
                    who_won = -1 if network_value_output[0][0] < 0 else 1
                    #print('who_won:', who_won)
                    #print('player ', game.player_turn)
                    #print('finished columns: ', game.finished_columns)
                    #print('won columns: ', game.player_won_column)
                    #game.board_game.print_board([])
                    is_over = True
                else:
                    who_won, is_over = game.is_finished()
            print('GAME', i ,'OVER - PLAYER', who_won, 'WON')
            print()
            if who_won == -1:
                victory_1 += 1
            else:
                victory_2 += 1
            # After the game is finished, we now know who won the game.
            # Therefore, save this info in all instances saved so far
            # for later use as input for the NN.
            for single_game in data_of_a_game:
                single_game.append(who_won)
            dataset_for_network.append(data_of_a_game)
        print('Player 1 won', victory_1,'time(s).')
        print('Player 2 won', victory_2,'time(s).')
        end = time.time()
        print('Time elapsed:', end - start)
        #
        # Training loop
        #
        print('TRAINING LOOP')

        #Get the channels of the states (Input for the NN)
        channels_input = [play[0] for game in dataset_for_network for play in game]
        channels_input = np.array(channels_input)
        channels_input = channels_input.reshape(channels_input.shape[0], channels_input.shape[2], channels_input.shape[3], -1)

        #print('shape channels_input')
        #print(channels_input.shape)

        #Get the probability distribution of the states (Label for the NN)
        dist_probs_label = [play[1] for game in dataset_for_network for play in game]
        dist_probs_label = [transform_dist_prob(dist_dict) for dist_dict in dist_probs_label]
        dist_probs_label = np.array(dist_probs_label)

        #print('shape dist_probs_label')
        #print(dist_probs_label.shape)

        #Get the distribution vector of the valid actions states (Input for the NN)
        valid_actions_dist_input = copy.copy(dist_probs_label)
        valid_actions_dist_input[valid_actions_dist_input > 0] = 1

        #print('shape valid action')
        #print(valid_actions_dist_input.shape)
        #print(valid_actions_dist_input)

        #Get the info of who won the games relating to the state (Label for the NN)
        who_won_label = [play[2] for game in dataset_for_network for play in game]
        who_won_label = np.array(who_won_label)
        who_won_label = np.expand_dims(who_won_label, axis=1)

        #print('shape who_won_label')
        #print(who_won_label.shape)

        x_train = [channels_input, valid_actions_dist_input]
        y_train = [dist_probs_label, who_won_label]

        #print('x_train shape: ', x_train.shape)
        #print('y_train shape: ', y_train.shape)

        model.fit(x_train, y_train, epochs=100)


    #results = model.evaluate(x_train, y_train, batch_size=4)
    #print('test loss, test acc:', results)

if __name__ == "__main__":
    main()