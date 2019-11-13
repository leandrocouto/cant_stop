from collections import defaultdict
import random
import time
import numpy as np
import copy
import random
from game import Board, Game
from utils import valid_positions_channel, finished_columns_channels
from utils import player_won_column_channels, player_turn_channel
from uct import MCTS
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import regularizers
from keras.layers import concatenate


def alphazero_loss_function(mcts_dist_prob, network_dist_prob):
    '''Custom loss function used in the AlphaGo Zero paper.'''
    def custom_loss(y_true, y_pred):
        mse = K.mean((y_true - y_pred)**2, keepdims=True)
        cross_entropy = np.dot(mcts_dist_prob.T, network_dist_prob) 
        return mse - cross_entropy
    return custom_loss

def define_model(config, mcts_dist_prob, network_dist_prob):
    '''Neural Network model implementation using Keras + Tensorflow'''
    #Valid positions channel
    state1 = Input(shape=(5,5,1))
    conv1 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(state1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    flat1 = Flatten()(pool1)
    #Finished columns channel 1
    state2 = Input(shape=(5,5,1))
    conv2 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(state2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat2 = Flatten()(pool2)
    #Finished columns channel 2
    state3 = Input(shape=(5,5,1))
    conv3 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(state3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    flat3 = Flatten()(pool3)
    #Player won columns channel 1
    state4 = Input(shape=(5,5,1))
    conv4 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(state4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    flat4 = Flatten()(pool4)
    #Player won columns channel 2
    state5 = Input(shape=(5,5,1))
    conv5 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(state5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    flat5 = Flatten()(pool5)
    #Player turn channel
    state6 = Input(shape=(5,5,1))
    conv6 = Conv2D(filters=10, kernel_size=2, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(state6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    flat6 = Flatten()(pool6)
    # merge
    merged = concatenate([flat1, flat2, flat3, flat4, flat5, flat6])
    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(merged)
    hidden_fc_prob_dist_2 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(27, kernel_regularizer=regularizers.l2(config.reg), activation='softmax')(hidden_fc_prob_dist_2)
    #Value of a state
    hidden_fc_value_1 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(merged)
    hidden_fc_value_2 = Dense(100, kernel_regularizer=regularizers.l2(config.reg), activation='relu')(hidden_fc_value_1)
    output_value = Dense(1, kernel_regularizer=regularizers.l2(config.reg), activation='tanh')(hidden_fc_value_2)

    model = Model(inputs=[state1, state2, state3, state4, state5, state6], 
                    outputs=[output_prob_dist, output_value])

    model.compile(loss=alphazero_loss_function(mcts_dist_prob, network_dist_prob), optimizer='adam')
    
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
    config = Config(c =1, n_simulations = 10, n_games = 50, n_players = 2, 
                    dice_number = 4, dice_value = 3, column_range = [2,6], 
                    offset = 2, initial_height = 1, mini_batch = 4, 
                    sample_size = 100, n_games_evaluate= 50, victory_rate = .55,
                    alphazero_iterations = 1, reg = 0.0001)
    #Neural network specification
    model = define_model(config, np.ndarray(shape=(27,1)), np.ndarray(shape=(27,1)))
    # summarize layers
    print(model.summary())
    #
    # Main loop of the algorithm
    #
    for _ in range(config.alphazero_iterations):
        dataset_for_network = []
        start = time.time()
        #
        # Self-play
        #
        for i in range(config.n_games):
            data_of_a_game = []
            game = Game(config)
            is_over = False
            uct = MCTS(config)
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
                #print('LIST OF CHANNELS')
                #print(single_game[0])
                #print('DIST PROB')
                #print(single_game[1])
                #print('Z')
                #print(single_game[2])
            #print(data_of_a_game)
            dataset_for_network.append(data_of_a_game)
        print('Player 1 won', victory_1,'time(s).')
        print('Player 2 won', victory_2,'time(s).')
        end = time.time()
        print('Time elapsed:', end - start)
        #
        # Training loop
        #
        channels = [play[0] for game in dataset_for_network for play in game]
        dist_probs = [play[1] for game in dataset_for_network for play in game]
        labels = [play[2] for game in dataset_for_network for play in game]

        channel_1 = [channel[0] for channel in channels]
        channel_2 = [channel[1] for channel in channels]
        channel_3 = [channel[2] for channel in channels]
        channel_4 = [channel[3] for channel in channels]
        channel_5 = [channel[4] for channel in channels]
        channel_6 = [channel[5] for channel in channels]
        
        x_train = [channel_1, channel_2, channel_3, channel_4, channel_5, channel_6]
        y_train = [dist_probs, labels]
        model.fit(x_train, y_train, nb_epoch=1000, batch_size=config.mini_batch)

if __name__ == "__main__":
    main()