import math, random
import keras.backend as K
import numpy as np
import copy
import collections
from random import sample
from players.uct_player import UCTPlayer, Node
from abc import abstractmethod
from collections import defaultdict
import time

class AlphaZeroPlayer(UCTPlayer):
    @abstractmethod
    def __init__(self, c, n_simulations, column_range, 
                        offset, initial_height, dice_value, network):
        """
        - column_range is a list denoting the range of the board game columns.
        - offset is the height difference between columns.
        - initial_height is the height of the columns at the board border.
        - dice_value is the number of faces of a single die.
        - network is the neural network AlphaZero trains.
        """

        super().__init__(c, n_simulations)
        self.column_range = column_range 
        self.offset = offset
        self.initial_height = initial_height
        self.dice_value = dice_value
        self.network = network

    @abstractmethod
    def rollout(self, node, scratch_game):
        """
        Return the game value of the end of the current simulation.
        Concrete classes must implement this method.
        """
        pass

    @abstractmethod
    def clone(self, reg, conv_number):
        """
        Clone self player.
        Tensorflow 1.3 complains about deepcopying keras models 
        while TF 2.0 > does not. Using this method in order to be able
        to run in either versions.
        Concrete classes must implement this method.
        """
        pass

    def select_action(self, game, root, dist_probability):
        """Return the action sampled from the distribution probability."""

        # Transform the dist. prob. dict. to a list of tuples.
        dist = [tuple(reversed(x)) for x in dist_probability.items()]
        # Create a list of additive probabilities in order to facilitate
        # the selection of the pair.
        additive_probabilities = []
        partial_sum = 0
        for i in range(len(dist)):
            partial_sum += dist[i][0]
            additive_probabilities.append(partial_sum)
        random_number = random.uniform(0.0, 1.0)
        selected_action_index = -1
        #Iterate through the list to get the sampled action.
        for i in range(len(additive_probabilities)):
            if random_number <= additive_probabilities[i]:
                selected_action_index = i
                break
        return dist[selected_action_index][1]

    def expand_children(self, parent):
        """Expand the children of the "parent" node."""
        
        # parent might not have any children given the dice configuration. 
        # Therefore, it should be checked if the player is busted. 
        # is_player_busted() automatically change the game dynamics 
        #(player turn, etc) if the player is indeed busted.
        is_busted = True
        while is_busted:
            is_busted = parent.state.is_player_busted(
                                parent.state.available_moves()
                                )
        valid_actions = parent.state.available_moves()
        
        for action in valid_actions:
            child_game = parent.state.clone()
            child_game.play(action)
            child_state = Node(child_game, parent)
            child_state.action_taken = action
            parent.children[action] = child_state

        #Update the  distribution probability of the children (node.p_a)
        network_input_parent = self.transform_to_input(parent.state, 
                                    self.column_range, self.offset, 
                                    self.initial_height
                                    )
        valid_actions_dist = self.transform_actions_to_dist(valid_actions)
        dist_prob, _= self.network.predict(
                                [network_input_parent, valid_actions_dist], 
                                batch_size = 1
                                )
        dist_prob = self.remove_invalid_actions(dist_prob[0], 
                                        parent.children.keys()
                                        )
        self.add_dist_prob_to_children(parent, dist_prob)

    def calculate_ucb_max(self, node, action):
        """
        Return the node modified PUCT (see AZ paper) value of the MAX player.
        """

        return node.q_a[action] + self.c * node.p_a[action] \
                * np.divide(math.sqrt(math.log(node.n_visits)), \
                            node.n_a[action]
                            )

    def calculate_ucb_min(self, node, action):
        """
        Return the node modified PUCT (see AZ paper) value of the MIN player.
        """

        return node.q_a[action] - self.c * node.p_a[action] \
                * np.divide(math.sqrt(math.log(node.n_visits)), \
                            node.n_a[action]
                            )

    def add_dist_prob_to_children(self, node, dist_prob):
        """
        Add the probability distribution given from the network to 
        node's children. This probability is used later to calculate
        the selection phase of the UCT algorithm.
        """

        standard_dist = self.get_standard_dist_with_id()
        standard_dist = collections.OrderedDict(standard_dist)
        for key in node.children.keys():
            node.p_a[key] = dist_prob[standard_dist[key]]

    def valid_positions_channel(self, column_range, offset, initial_height):
        """
        Return a channel that fills with a value of 1 if that cell is valid 
        and 0 otherwise.
        """

        rows = column_range[1] - column_range[0] + 1
        columns = initial_height + offset * (rows//2)
        channel = np.zeros((rows, columns), dtype=int)
        height = initial_height
        for i in range(rows):
            for j in range(height):
                channel[i][j] = 1
            if i < rows//2:
                height += offset
            else:
                height -= offset
        return channel

    def finished_columns_channels(self, state, channel):
        """
        Return two channels that fills with a 1 in a column if the respective 
        player has permanently won that column.
        """
        
        channel_player_1 = np.array(channel)
        channel_player_2 = np.array(channel)

        finished_cols_player_1 = [item[0] for item in state.finished_columns 
                                    if item[1] == 1
                                    ]
        finished_cols_player_2 = [item[0] for item in state.finished_columns 
                                    if item[1] == 2
                                    ]

        for i in range(channel_player_1.shape[0]):
            if i+2 not in finished_cols_player_1:
                for j in range(channel_player_1.shape[1]):
                    channel_player_1[i][j] = 0

        for i in range(channel_player_2.shape[0]):
            if i+2 not in finished_cols_player_2:
                for j in range(channel_player_2.shape[1]):
                    channel_player_2[i][j] = 0

        return channel_player_1, channel_player_2

    def player_won_column_channels(self, state, channel):
        """
        Return two channels that fills with a 1 in a column if the respective 
        player has temporarily won that column.
        """

        channel_player_1 = np.array(channel)
        channel_player_2 = np.array(channel)
        
        player_won_column_player_1 = [item[0] 
                                        for item in state.player_won_column 
                                        if item[1] == 1
                                        ]
        player_won_column_player_2 = [item[0] 
                                        for item in state.player_won_column 
                                        if item[1] == 2
                                        ]

        for i in range(channel_player_1.shape[0]):
            if i+2 not in player_won_column_player_1:
                for j in range(channel_player_1.shape[1]):
                    channel_player_1[i][j] = 0

        for i in range(channel_player_2.shape[0]):
            if i+2 not in player_won_column_player_2:
                for j in range(channel_player_2.shape[1]):
                    channel_player_2[i][j] = 0

        return channel_player_1, channel_player_2

    def player_turn_channel(self, state, channel):
        """
        Return a channel filled with 1 if it is the the player 1's turn and a 
        channel filled with 0 otherwise.
        """

        shape = channel.shape
        if state.player_turn == 1:
            return np.ones(shape, dtype=int)
        else:
            return np.zeros(shape, dtype=int)

    def transform_to_input(self, game, column_range, offset, initial_height):
        """
        Receive the game state and return the six channels used as input 
        for the network.
        """

        channel_valid = self.valid_positions_channel(column_range, offset, 
                                                        initial_height
                                                        )
        channel_finished_1, channel_finished_2 = \
                        self.finished_columns_channels(game, channel_valid)
        channel_won_column_1, channel_won_column_2 = \
                        self.player_won_column_channels(game, channel_valid)
        channel_turn = self.player_turn_channel(game, channel_valid)
        list_of_channels = [channel_valid, 
                            channel_finished_1, channel_finished_2,
                            channel_won_column_1, channel_won_column_2, 
                            channel_turn
                            ]
        list_of_channels = np.array(list_of_channels)
        list_of_channels = np.expand_dims(list_of_channels, axis = 0)
        return list_of_channels

    def remove_invalid_actions(self, dist_prob, keys):
        """Re-normalize the distribution based only on the valid actions."""

        standard_dist = self.get_standard_dist_with_id()
        standard_dist = collections.OrderedDict(standard_dist)
        indexes_of_dist = [standard_dist[key] for key in keys 
                            if key in standard_dist
                            ]
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

    def transform_dist_prob(self, dist_prob):
        """
        Transform a list of the only possible action probability distributions
        into a fully distribution (with all the invalid actions). 
        This is needed for the loss function that expects full distributions 
        on all actions.
        """

        standard_dist = self.get_zeroed_standard_dist()
        complete_dict = collections.OrderedDict(standard_dist)
        for key, value in complete_dict.items():
            for key_2, value_2 in dist_prob.items():
                if key == key_2:
                    complete_dict[key] += dist_prob[key_2]
        complete_dict = [value for _, value in complete_dict.items()]
        return complete_dict

    def transform_actions_to_dist(self, actions):
        """
        Given a list of possible actions in a certain state, return the 
        distribution probability over all actions. 
        Write 1 if an action is valid, 0 otherwise.
        """

        standard_dist = self.get_standard_dist_with_id()
        standard_dist = collections.OrderedDict(standard_dist)

        valid_actions_dist = np.zeros(len(standard_dist))

        for action in actions:
            valid_actions_dist[standard_dist[action]] = 1

        valid_actions_dist = np.expand_dims(valid_actions_dist, axis = 0)
        
        return valid_actions_dist

    def transform_dataset_to_input(self, dataset_for_network):
        """
        Transform the dataset collected by the selfplay into
        separated training and label data used as input for
        the network.
        """

        #Get the channels of the states (Input)
        channels_input = [play[0] for game in dataset_for_network 
                            for play in game
                            ]
        channels_input = np.array(channels_input)

        #Get the probability distribution of the states (Label)
        dist_probs_label = [play[1] for game in dataset_for_network 
                            for play in game
                            ]
        dist_probs_label = [self.transform_dist_prob(dist_dict) 
                            for dist_dict in dist_probs_label
                            ]
        dist_probs_label = np.array(dist_probs_label)

        #Get the distribution vector of the valid actions states (Input)
        valid_actions_dist_input = copy.copy(dist_probs_label)
        valid_actions_dist_input[valid_actions_dist_input > 0] = 1

        #Get the info of who won the games relating to the state (Label)
        who_won_label = [play[2] for game in dataset_for_network 
                            for play in game
                            ]
        who_won_label = np.array(who_won_label)
        who_won_label = np.expand_dims(who_won_label, axis=1)

        return channels_input, valid_actions_dist_input, \
                dist_probs_label, who_won_label

    def sample_input(self, channels_input, valid_actions_dist_input, 
                        dist_probs_label, who_won_label, mini_batch):
        """Sample mini_batch inputs from the whole collected dataset."""

        # If the mini_batch size is lesser or equal than the data itself, then
        # sample accordingly
        if mini_batch <= channels_input.shape[0]:
            # indexes of the whole input to be sampled

            index_list = sample(range(channels_input.shape[0]), 
                                channels_input.shape[0] - mini_batch
                                )

            channels_input = np.delete(channels_input, index_list, 0)
            valid_actions_dist_input = np.delete(valid_actions_dist_input, 
                                                index_list, 0
                                                )
            dist_probs_label = np.delete(dist_probs_label, index_list, 0)
            who_won_label = np.delete(who_won_label, index_list, 0)

        x_train = [channels_input, valid_actions_dist_input]
        y_train = [dist_probs_label, who_won_label]

        return x_train, y_train

    def get_standard_dist_with_id(self):
        """Return a list with all possible actions with an ID to it."""
        
        standard_dist = []

        action_id = 0
        for i in range(2, self.dice_value * 2 + 1):
            for j in range(i, self.dice_value * 2 + 1):
                standard_dist.append(((i, j), action_id))
                action_id += 1

        for i in range(2, self.dice_value * 2 + 1):
            standard_dist.append(((i,), action_id))
            action_id += 1

        standard_dist.append((('y'), action_id))
        action_id += 1
        standard_dist.append((('n'), action_id))

        return standard_dist

    def get_zeroed_standard_dist(self):
        """Return a list with all possible actions with an ID = 0 to it."""
        
        standard_dist = []

        for i in range(2, self.dice_value * 2 + 1):
            for j in range(i, self.dice_value * 2 + 1):
                standard_dist.append(((i, j), 0))

        for i in range(2, self.dice_value * 2 + 1):
            standard_dist.append(((i,), 0))

        standard_dist.append((('y'), 0))
        standard_dist.append((('n'), 0))


        return standard_dist