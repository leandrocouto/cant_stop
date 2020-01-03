import copy
import tensorflow as tf
import collections
import os

class Config:
    """ General configuration class for the game board, UCT and NN """
    def __init__(self, c, n_simulations, n_games, n_games_evaluate, 
                    maximum_game_length, n_players, dice_number, dice_value, 
                    column_range, offset, initial_height, mini_batch, sample_size,
                    victory_rate, alphazero_iterations, reg, epochs, conv_number,
                    use_playout):
        """
        - c is the constant the balance exploration and exploitation.
        - n_simulations is the number of simulations made in the UCT algorithm.
        - n_games is the number of games played in the self-play scheme.
        - n_games_evaluate is the number of games played to evaluate the current
          network against the previous one.
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
        - victory_rate is the % of victories necessary for the new network to
          overwrite the previously one.
        - alphazero_iterations is the total number of iterations of the learning
          algorithm: selfplay -> training loop -> evaluate network (repeat).
        - reg is the L2 regularization parameter.
        - epochs is the number of training epochs.
        - conv_number is the number of convolutional layers in the network.
        - use_playout is a boolean that allows the program to also calculate the 
          MSE playout in the network (besides using the network to estimate who
          won the game). Used for result analysis.
        """
        self.c = c
        self.n_simulations = n_simulations
        self.n_games = n_games
        self.n_games_evaluate = n_games_evaluate
        self.maximum_game_length = maximum_game_length
        self.n_players = n_players
        self.dice_number = dice_number
        self.dice_value = dice_value
        self.column_range = column_range 
        self.offset = offset
        self.initial_height = initial_height
        self.mini_batch = mini_batch
        self.sample_size = sample_size
        self.victory_rate = victory_rate
        self.alphazero_iterations = alphazero_iterations
        self.reg = reg
        self.epochs = epochs
        self.conv_number = conv_number
        self.use_playout = use_playout

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
    list_of_channels = tf.cast(list_of_channels, tf.float64)
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

def transform_dist_prob(dist_prob):
    """
    Transform a list of the only possible action probability distributions
    into a fully distribution (with all the invalid actions). This is needed
    for the loss function that expects full distributions on all actions.
    """
    standard_dist = [((2,2),0), ((2,3),0), ((2,4),0), ((2,5),0), ((2,6),0),
                     ((3,2),0), ((3,3),0), ((3,4),0), ((3,5),0), ((3,6),0),
                     ((4,2),0), ((4,3),0), ((4,4),0), ((4,5),0), ((4,6),0),
                     ((5,2),0), ((5,3),0), ((5,4),0), ((5,5),0), ((5,6),0),
                     ((6,2),0), ((6,3),0), ((6,4),0), ((6,5),0), ((6,6),0),
                     ((2,),0), ((3,),0), ((4,),0), ((5,),0), ((6,),0),
                    ('y',0), ('n',0)]
    complete_dict = collections.OrderedDict(standard_dist)
    for key, value in complete_dict.items():
        for key_2, value_2 in dist_prob.items():
            if key == key_2:
                complete_dict[key] += dist_prob[key_2]
    complete_dict = [value for _, value in complete_dict.items()]
    return complete_dict

def transform_actions_to_dist(actions):
    """
    Given a list of possible actions in a certain state, return the dstribution
    probability over all actions. Writes 1 if an action is valid, 0 otherwise.
    """
    standard_dist = [((2,2),0), ((2,3),1), ((2,4),2), ((2,5),3), ((2,6),4),
                     ((3,2),5), ((3,3),6), ((3,4),7), ((3,5),8), ((3,6),9),
                     ((4,2),10), ((4,3),11), ((4,4),12), ((4,5),13), ((4,6),14),
                     ((5,2),15), ((5,3),16), ((5,4),17), ((5,5),18), ((5,6),19),
                     ((6,2),20), ((6,3),21), ((6,4),22), ((6,5),23), ((6,6),24),
                     ((2,),25), ((3,),26), ((4,),27), ((5,),28), ((6,),29),
                    ('y',30), ('n',31)]
    standard_dist = collections.OrderedDict(standard_dist)

    valid_actions_dist = np.zeros(32)

    for action in actions:
        valid_actions_dist[standard_dist[action]] = 1

    valid_actions_dist = np.expand_dims(valid_actions_dist, axis=0)
    
    return valid_actions_dist

def transform_dataset_to_input(dataset_for_network):
    """
    Transform the dataset collected by the selfplay into
    separated training and label data used as input for
    the network.
    """
    
    #Get the channels of the states (Input for the NN)
    channels_input = [play[0] for game in dataset_for_network for play in game]
    channels_input = np.array(channels_input)
    channels_input = channels_input.reshape(channels_input.shape[0], channels_input.shape[2], channels_input.shape[3], -1)

    #Get the probability distribution of the states (Label for the NN)
    dist_probs_label = [play[1] for game in dataset_for_network for play in game]
    dist_probs_label = [transform_dist_prob(dist_dict) for dist_dict in dist_probs_label]
    dist_probs_label = np.array(dist_probs_label)

    #Get the distribution vector of the valid actions states (Input for the NN)
    valid_actions_dist_input = copy.copy(dist_probs_label)
    valid_actions_dist_input[valid_actions_dist_input > 0] = 1

    #Get the info of who won the games relating to the state (Label for the NN)
    who_won_label = [play[2] for game in dataset_for_network for play in game]
    who_won_label = np.array(who_won_label)
    who_won_label = np.expand_dims(who_won_label, axis=1)

    x_train = [channels_input, valid_actions_dist_input]
    y_train = [dist_probs_label, who_won_label]

    return x_train, y_train