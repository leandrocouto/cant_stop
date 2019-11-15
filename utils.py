import copy
import numpy as np

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
