import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import tensorflow as tf
import collections
import os


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

def generate_graphs(data_for_analysis, data_net_vs_net, config, model):
    """Generate graphs based on alphazero iterations."""
    save_path = 'graphs_'+str(config.n_simulations)+'_'+str(config.n_games) \
                + '_' + str(config.alphazero_iterations) + '_' + str(config.conv_number) +  '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Data preparation

    total_loss = []
    output_dist_loss_history = []
    output_value_loss_history = []
    dist_metric_history = [] 
    value_metric_history = [] 
    victory_1 = [] 
    victory_2 = [] 
    loss_eval = [] 
    dist_loss_eval = [] 
    value_loss_eval = [] 
    dist_metric_eval = [] 
    value_metric_eval = [] 
    victory_1_eval = [] 
    victory_2_eval = []
    victory_1_eval_net = [] 
    victory_2_eval_net = []
    for analysis in data_for_analysis:
        total_loss.append(analysis[0])
        output_dist_loss_history.append(analysis[1])
        output_value_loss_history.append(analysis[2])
        dist_metric_history.append(analysis[3])
        value_metric_history.append(analysis[4])
        victory_1.append(analysis[5])
        victory_2.append(analysis[6])
        loss_eval.append(analysis[7])
        dist_loss_eval.append(analysis[8])
        value_loss_eval.append(analysis[9])
        dist_metric_eval.append(analysis[10])
        value_metric_eval.append(analysis[11])
        victory_1_eval.append(analysis[12])
        victory_2_eval.append(analysis[13])
    for analysis in data_net_vs_net:
        victory_1_eval_net.append(analysis[0])
        victory_2_eval_net.append(analysis[1])

    # Total loss
    x = np.array(range(1, len(total_loss) + 1))
    y = np.array(total_loss)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Loss', 
        title='Total loss in training')
    ax.grid()
    fig.savefig(save_path + "1_total_loss.png")

    # Probability distribution loss
    x = np.array(range(1, len(output_dist_loss_history) + 1))
    y = np.array(output_dist_loss_history)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Loss',
            title='Probability distribution loss in training')
    ax.grid()
    fig.savefig(save_path + "2_output_dist_loss_history.png")

    # Value loss
    x = np.array(range(1, len(output_value_loss_history) + 1))
    y = np.array(output_value_loss_history)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Loss', 
            title='Value loss in training')
    ax.grid()
    fig.savefig(save_path + "3_output_value_loss_history.png")

    # Probability Distribution CE error
    x = np.array(range(1, len(dist_metric_history) + 1))
    y = np.array(dist_metric_history)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Cross entropy error',
            title='Probability Distribution CE error in training')
    ax.grid()
    fig.savefig(save_path + "4_dist_metric_history.png")

    # Value MSE error
    x = np.array(range(1, len(value_metric_history) + 1))
    y = np.array(value_metric_history)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='MSE error',
            title='Value MSE error in training')
    ax.grid()
    fig.savefig(save_path + "5_value_metric_history.png")

    # Victory of player 1 in training
    x = np.array(range(1, len(victory_1) + 1))
    y = np.array(victory_1)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Number of victories',
            title='Victory of player 1 in training')
    ax.grid()
    fig.savefig(save_path + "6_victory_1.png")

    # Victory of player 2 in training
    x = np.array(range(1, len(victory_2) + 1))
    y = np.array(victory_2)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Number of victories',
            title='Victory of player 2 in training')
    ax.grid()
    fig.savefig(save_path + "7_victory_2.png")

    # Total loss in evaluation
    x = np.array(range(1, len(loss_eval) + 1))
    y = np.array(loss_eval)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Loss',
            title='Total loss in evaluation')
    ax.grid()
    fig.savefig(save_path + "8_loss_eval.png")

    # Probability distribution loss in evaluation
    x = np.array(range(1, len(dist_loss_eval) + 1))
    y = np.array(dist_loss_eval)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Loss',
            title='Probability distribution loss in evaluation')
    ax.grid()
    fig.savefig(save_path + "9_dist_loss_eval.png")

    # Value loss in evaluation
    x = np.array(range(1, len(value_loss_eval) + 1))
    y = np.array(value_loss_eval)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Loss',
            title='Value loss in evaluation')
    ax.grid()
    fig.savefig(save_path + "10_value_loss_eval.png")

    # Probability Distribution CE error in evaluation
    x = np.array(range(1, len(dist_metric_history) + 1))
    y = np.array(dist_metric_history)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Crossentropy error',
            title='Probability Distribution CE error in evaluation')
    ax.grid()
    fig.savefig(save_path + "11_dist_metric_history.png")

    # Value MSE error in evaluation
    x = np.array(range(1, len(value_metric_eval) + 1))
    y = np.array(value_metric_eval)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='MSE error',
            title='Value MSE error in evaluation')
    ax.grid()
    fig.savefig(save_path + "12_value_metric_eval.png")

    # Victory of player 1 in evaluation
    x = np.array(range(1, len(victory_1_eval) + 1))
    y = np.array(victory_1_eval)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Number of victories',
            title='Victory of player 1 in evaluation - Net vs. UCT')
    ax.grid()
    fig.savefig(save_path + "13_victory_1_eval.png")

    # Victory of player 2 in evaluation
    x = np.array(range(1, len(victory_2_eval) + 1))
    y = np.array(victory_2_eval)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Iterations', ylabel='Number of victories',
            title='Victory of player 2 in evaluation - Net vs. UCT')
    ax.grid()
    fig.savefig(save_path + "14_victory_2_eval.png")

    # This can happen if the new network always lose to the
    # previous one.
    if len(victory_1_eval_net) != 0:
        # Victory of player 1 in evaluation - net vs net
        x = np.array(range(1, len(data_net_vs_net) + 1))
        y = np.array(victory_1_eval_net)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Number of victories',
                title='Victory of player 1 in evaluation - Net vs. Net')
        ax.grid()
        fig.savefig(save_path + "15_victory_1_eval_net_vs_net.png")

        # Victory of player 2 in evaluation - net vs net
        x = np.array(range(1, len(data_net_vs_net) + 1))
        y = np.array(victory_2_eval_net)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Number of victories',
                title='Victory of player 2 in evaluation - Net vs. Net')
        ax.grid()
        fig.savefig(save_path + "16_victory_2_eval_net_vs_net.png")

    # Save the model
    model.save(save_path + 'model.h5')





