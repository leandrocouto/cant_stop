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
from utils import transform_dist_prob
from uct import MCTS
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import regularizers
from collections import Counter

def define_model(config):
    """Neural Network model implementation using Keras + Tensorflow."""
    state = Input(shape = (5,5,6), name='States_Channels_Input')
    dist_prob = Input(shape = (1,32), name='Distribution_Probabilities_Input') 
    labels = Input(shape = (1,), name='Labels_Input')

    print('shape state', state.shape)
    print('shape dist', dist_prob.shape)
    print('shape labels', labels.shape)

    conv = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='Conv_Layer')(state)
    pool = MaxPooling2D(pool_size=(2, 2), name='Pooling_Layer')(conv)
    flat = Flatten(name='Flatten_Layer')(pool)
    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='FC_Prob_1')(flat)
    hidden_fc_prob_dist_2 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='FC_Prob_2')(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(32, kernel_regularizer=regularizers.l2(config.reg), activation='softmax', name='Output_Dist')(hidden_fc_prob_dist_2)
    #Value of a state
    hidden_fc_value_1 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='FC_Value_1')(flat)
    hidden_fc_value_2 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu', name='FC_Value_2')(hidden_fc_value_1)
    output_value = Dense(1, kernel_regularizer=regularizers.l2(config.reg), activation='tanh', name='Output_Value')(hidden_fc_value_2)

    mse_loss = K.mean(K.square(output_value - labels), axis=-1)
    print('mse shape', mse_loss.shape)
    cross_entropy_loss = K.dot(K.transpose(output_prob_dist), output_prob_dist)
    custom_loss = mse_loss - cross_entropy_loss
    print('custom_loss shape', custom_loss.shape)

    model = Model(inputs=[state, dist_prob, labels], outputs=[output_prob_dist, output_value])


    model.add_loss(custom_loss)

    model.compile(optimizer='adam', metrics=['accuracy'])
    
    return model
    
class Config:
    """ General configuration class for the game board, UCT and NN """
    def __init__(self, c, n_simulations, n_games, n_players, dice_number,
                    dice_value, column_range, offset, initial_height,
                    mini_batch, sample_size, n_games_evaluate, victory_rate,
                    alphazero_iterations, reg):
        """
        - c is the constant the balance exploration and exploitation.
        - n_simulations is the number of simulations made in the UCT algorithm.
        - n_games is the number of games played in the self-play scheme.
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
    config = Config(c = 1, n_simulations = 10, n_games = 2, n_players = 2, 
                    dice_number = 4, dice_value = 3, column_range = [2,6], 
                    offset = 2, initial_height = 1, mini_batch = 2, 
                    sample_size = 100, n_games_evaluate= 50, victory_rate = .55,
                    alphazero_iterations = 1, reg = 0.0001)
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
            while not is_over:
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
                    if game.player_turn == 1:
                        chosen_play, dist_probability = uct.run_mcts(game)
                    else:
                        chosen_play, dist_probability = uct.run_mcts(game)
                    current_play = [list_of_channels, dist_probability]
                    data_of_a_game.append(current_play)
                    game.play(chosen_play)
                who_won, is_over = game.is_finished()
            print()
            print('GAME', i ,'OVER - PLAYER', who_won, 'WON')
            if who_won == -1:
                victory_1 += 1
            else:
                victory_2 += 1
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
        channels = [play[0] for game in dataset_for_network for play in game]
        channels = np.array(channels)
        channels = channels.reshape(channels.shape[0], channels.shape[2], channels.shape[3], -1)

        print('shape channels')
        print(channels.shape)

        dist_probs = [play[1] for game in dataset_for_network for play in game]
        dist_probs = [transform_dist_prob(dist_dict) for dist_dict in dist_probs]
        dist_probs = np.array(dist_probs)

        print('shape dist')
        print(dist_probs.shape)

        labels = [play[2] for game in dataset_for_network for play in game]
        labels = np.array(labels)

        print('shape label')
        print(labels.shape)

        x_train = [channels, dist_probs, labels]
        y_train = [dist_probs, labels]

        model.fit(x_train, y_train, epochs=10)


    results = model.evaluate(x_train, y_train, batch_size=4)
    print('test loss, test acc:', results)

if __name__ == "__main__":
    main()