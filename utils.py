import copy
import numpy as np
import collections


def valid_positions_channel(config):
    """
    Return a channel that fills with a value of 1 if that cell is valid and
    0 otherwise.
    """
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
    """
    Return two channels that fills with a 1 in a column if the respective player
    has permanently won that column.
    """
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
    """
    Return two channels that fills with a 1 in a column if the respective player
    has temporarily won that column.
    """
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
    """
    Return a channel filled with 1 if it is the the player 1's turn and a channel
    filled with 0 otherwise.
    """
    shape = channel.shape
    if state.player_turn == 1:
        return np.ones(shape, dtype=int)
    else:
        return np.zeros(shape, dtype=int)

def transform_to_input(game, config):
    """Receive the game state and return the six channels used as input for the network"""
    channel_valid = valid_positions_channel(config)
    channel_finished_1, channel_finished_2 = finished_columns_channels(game, channel_valid)
    channel_won_column_1, channel_won_column_2 = player_won_column_channels(game, channel_valid)
    channel_turn = player_turn_channel(game, channel_valid)
    list_of_channels = [channel_valid, channel_finished_1, channel_finished_2,
                        channel_won_column_1, channel_won_column_2, channel_turn]
    list_of_channels = np.array(list_of_channels)
    list_of_channels = np.expand_dims(list_of_channels, axis=0)
    list_of_channels = list_of_channels.reshape(list_of_channels.shape[0], 
                        list_of_channels.shape[2], list_of_channels.shape[3], -1)
    return list_of_channels

def remove_invalid_actions(dist_prob, keys):
    """Re-normalize the distribution based only on the valid actions."""
    standard_dist = [((2,2),0), ((2,3),1), ((2,4),2), ((2,5),3), ((2,6),4),
                     ((3,2),5), ((3,3),6), ((3,4),7), ((3,5),8), ((3,6),9),
                     ((4,2),10), ((4,3),11), ((4,4),12), ((4,5),13), ((4,6),14),
                     ((5,2),15), ((5,3),16), ((5,4),17), ((5,5),18), ((5,6),19),
                     ((6,2),20), ((6,3),21), ((6,4),22), ((6,5),23), ((6,6),24),
                     ((2,),25), ((3,),26), ((4,),27), ((5,),28), ((6,),29),
                    ('y',30), ('n',31)]
    standard_dist = collections.OrderedDict(standard_dist)
    indexes_of_dist = [standard_dist[key] for key in keys if key in standard_dist]
    sum_not_valid = 0
    sum_valid = 0

    for i in range(len(dist_prob)):
        if i in indexes_of_dist:
            sum_valid += dist_prob[i]
        else:
            sum_not_valid += dist_prob[i]
    for i in range(len(dist_prob)):
        if i in indexes_of_dist:
            dist_prob[i] += sum_not_valid / (sum_valid / dist_prob[i])
        else:
            dist_prob[i] = 0.

    return dist_prob



