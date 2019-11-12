from collections import defaultdict
import random
from game import Board, Game
from uct import MCTS
import time
import numpy as np
import copy
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def valid_positions_channel(config):
    rows = config.column_range[1] - config.column_range[0] + 1
    columns = config.initial_height + config.offset * (rows//2)
    channel = np.zeros((rows, columns), dtype=int)
    height = config.initial_height
    for i in range(rows):
        for j in range(height):
            channel[i][j] = 1
        if i < rows//2:
            height += config.offset
        else:
            height -= config.offset
    return channel

def finished_columns_channels(state, channel):
    channel_player_1 = copy.deepcopy(channel)
    channel_player_2 = copy.deepcopy(channel)
    finished_columns_player_1 = [item[0] for item in state.finished_columns if item[1] == 1]
    finished_columns_player_2 = [item[0] for item in state.finished_columns if item[1] == 2]

    for i in range(channel_player_1.shape[0]):
        if i+2 not in finished_columns_player_1:
            for j in range(channel_player_1.shape[1]):
                channel_player_1[i][j] = 0

    for i in range(channel_player_2.shape[0]):
        if i+2 not in finished_columns_player_2:
            for j in range(channel_player_2.shape[1]):
                channel_player_2[i][j] = 0

    return channel_player_1, channel_player_2

def player_won_column_channels(state, channel):
    channel_player_1 = copy.deepcopy(channel)
    channel_player_2 = copy.deepcopy(channel)
    player_won_column_player_1 = [item[0] for item in state.player_won_column if item[1] == 1]
    player_won_column_player_2 = [item[0] for item in state.player_won_column if item[1] == 2]

    for i in range(channel_player_1.shape[0]):
        if i+2 not in player_won_column_player_1:
            for j in range(channel_player_1.shape[1]):
                channel_player_1[i][j] = 0

    for i in range(channel_player_2.shape[0]):
        if i+2 not in player_won_column_player_2:
            for j in range(channel_player_2.shape[1]):
                channel_player_2[i][j] = 0

    return channel_player_1, channel_player_2

def player_turn_channel(state, channel):
    shape = channel.shape
    if state.player_turn == 1:
        return np.ones(shape, dtype=int)
    else:
        return np.zeros(shape, dtype=int)
class Config:
    """ General configuration class for the game board, UCT and NN """
    def __init__(self, c, n_simulations, n_games, n_players, dice_number,
                    dice_value, column_range, offset, initial_height):
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

def main():
    victory_1 = 0
    victory_2 = 0
    config = Config(c =1, n_simulations = 10, n_games = 10, n_players = 2, 
                    dice_number = 4, dice_value = 3, column_range = [2,6], 
                    offset = 2, initial_height = 1)
    #Neural network specification
    state = Input(shape=(5,5,6))
    conv = Conv2D(filters=10, kernel_size=2, activation='relu')(state)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    flat = Flatten()(pool)
    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(100, activation='relu')(flat)
    hidden_fc_prob_dist_2 = Dense(100, activation='relu')(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(27, activation='sigmoid')(hidden_fc_prob_dist_2)
    #Value of a state
    hidden_fc_value_1 = Dense(100, activation='relu')(flat)
    hidden_fc_value_2 = Dense(100, activation='relu')(hidden_fc_value_1)
    output_value = Dense(1, activation='sigmoid')(hidden_fc_value_2)

    model = Model(inputs=state, outputs=[output_prob_dist, output_value])
    # summarize layers
    print(model.summary())
    
    dataset_for_network = []
    start = time.time()
    for i in range(config.n_games):
        data_of_a_game = []
        game = Game(config)
        is_over = False
        uct = MCTS(config)
        print('Game', i, 'has started.')
        while not is_over:
            channel_valid = valid_positions_channel(config)
            print(channel_valid)
            channel_finished_1, channel_finished_2 = finished_columns_channels(game, channel_valid)
            channel_won_column_1, channel_won_column_2 = player_won_column_channels(game, channel_valid)
            channel_turn = player_turn_channel(game, channel_valid)
            print(channel_turn)
            moves = game.available_moves()
            print(moves)
            if game.is_player_busted(moves):
                continue
            else:
                if game.player_turn == 1:
                    chosen_play, dist_probability = uct.run_mcts(game)
                else:
                    chosen_play, dist_probability = uct.run_mcts(game)
                current_play = [game, dist_probability]
                data_of_a_game.append(current_play)
                game.play(chosen_play)
            who_won, is_over = game.is_finished()
        print()
        print('GAME', i ,'OVER - PLAYER', who_won, 'WON')
        if who_won == 1:
            victory_1 += 1
        else:
            victory_2 += 1
        for single_game in data_of_a_game:
            print(single_game)
            single_game.append(who_won)
        #print(data_of_a_game)
    print('Player 1 won', victory_1,'time(s).')
    print('Player 2 won', victory_2,'time(s).')
    end = time.time()
    print('Time elapsed:', end - start)

if __name__ == "__main__":
    main()