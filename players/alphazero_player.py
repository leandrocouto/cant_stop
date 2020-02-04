import math, random
import numpy as np
import copy
import collections
from players.player import Player
from abc import abstractmethod
from collections import defaultdict

class Node:
    def __init__(self, state, parent=None):
        """
        - n_visits is the number of visits in this node
        - n_a is a dictionary {key, value} where key is the action taken from 
          this node and value is the number of times this action was chosen.
        - q_a is a dictionary {key, value} where key is the action taken from
          this node and value is the mean reward of the simulation that passed
          through this node and used action 'a'.
        - p_a is a dictionary {key, value} where key is the action taken from
          this node and value is the probability given by the trained network
          to choose this action. Value updated in the expand_children() method
          from the MCTS class.
        - parent is a Node object. 'None' if this node is the root of the tree.
        - children is a dictionary {key, value} where key is the action 
          taken from this node and value is the resulting node after applying 
          the action.
        - action_taken is the action taken from parent to get to this node.
        """
        self.state = state
        self.n_visits = 0
        self.n_a = {}
        self.q_a = {}
        self.p_a = {}
        # These will initialize value = 0 for whichever keys yet to be added.
        self.n_a = defaultdict(lambda: 0, self.n_a)
        self.q_a = defaultdict(lambda: 0, self.q_a)
        self.p_a = defaultdict(lambda: 0, self.q_a)
        self.parent = parent
        self.children = {}
        self.action_taken = None


    def is_expanded(self):
        """Return a boolean."""
        return len(self.children) > 0

class AlphaZeroPlayer(Player):
    @abstractmethod
    def __init__(self, alphazero_config):
        self.root = None
        self.alphazero_config = alphazero_config
        self.action = None
        self.dist_probability = None

    @abstractmethod
    def expand_children(self, parent):
        """Expand the children of the "parent" node"""
        pass

    @abstractmethod
    def rollout(self, node, scratch_game):
        """
        Return the game value of the end of the current simulation.
        Concrete classes must implement this method.
        """
        pass

    @abstractmethod
    def calculate_ucb_max(self, node):
        """
        Return the node UCB value of the MAX player.
        Concrete classes must implement this method.
        """
        pass

    @abstractmethod
    def calculate_ucb_min(self, node):
        """
        Return the node UCB value of the MIN player.
        Concrete classes must implement this method.
        """
        pass

    def get_action(self, game):
        """ Return the action given by the UCT algorithm. """
        action, dist_probability = self.run_UCT(game)
        self.action = action
        self.dist_probability = dist_probability
        # Reset the tree for the future run_UCT call
        self.root = None
        return action

    def get_dist_probability(self):
        """ 
        Return the actions distribution probability regarding
        the game passed as parameter in get_action. Should be 
        called after get_action.
        """
        return self.dist_probability

    def run_UCT(self, game):
        """Main routine of the UCT algoritm."""
        if self.root == None:
            self.root = Node(game.clone())
        else:
            self.root.state = game.clone()
        #Expand the children of the root before the actual algorithm
        self.expand_children(self.root)

        for _ in range(self.alphazero_config.n_simulations):
            node = self.root
            scratch_game = game.clone()
            search_path = [node]
            while node.is_expanded():
                action, new_node = self.select_child(node)
                node.action_taken = action
                scratch_game.play(action)
                search_path.append(new_node)
                node = copy.deepcopy(new_node)
            #At this point, a leaf was reached.
            #If it was not visited yet, then perform the rollout and
            #backpropagates the reward returned from the end of the simulation.
            #If it has been visited, then expand its children, choose the one
            #with the highest ucb score and do a rollout from there.
            if node.n_visits == 0:
                rollout_value = self.rollout(node, scratch_game)
                self.backpropagate(search_path, action, rollout_value)
            else:
                self.expand_children(node)
                action_for_rollout, node_for_rollout = self.select_child(node)
                search_path.append(node)
                rollout_value = self.rollout(node_for_rollout, scratch_game)
                self.backpropagate(search_path, action_for_rollout, rollout_value)
        action = self.select_action(game, self.root)
        dist_probability = self.distribution_probability()
        self.root = self.root.children[action]
        return action, dist_probability    

    def select_action(self, game, root):
        """Return the action with the highest visit score."""
        visit_counts = [(child.n_visits, action)
                      for action, child in root.children.items()]
        # Sort based on the number of visits
        visit_counts.sort(key=lambda t: t[0])
        _, action = visit_counts[-1]
        return action

    def backpropagate(self, search_path, action, value):
        """Propagate the value from rollout all the way up the tree to the root."""
        for node in search_path:
            node.n_visits += 1
            node.n_a[node.action_taken] += 1 
            # Incremental mean calculation
            node.q_a[node.action_taken] = (node.q_a[node.action_taken] * 
                                            (node.n_visits - 1) + value) / \
                                                node.n_visits

    def select_child(self, node):
        """Return the child Node with the highest UCB score."""
        ucb_values = []
        for action, child in node.children.items():
            if node.state.player_turn == 1:
                if child.n_visits == 0:
                    ucb_max = float('inf')
                else:
                    ucb_max =  self.calculate_ucb_max(node, action)
                ucb_values.append((ucb_max, action, child))
            else:
                if child.n_visits == 0:
                    ucb_min = float('-inf')
                else:
                    ucb_min =  self.calculate_ucb_min(node, action)
                ucb_values.append((ucb_min, action, child))
        # Sort the list based on the ucb score
        ucb_values.sort(key=lambda t: t[0])
        if node.state.player_turn == 1:
            best_ucb, best_action, best_child = ucb_values[-1]
        else:
            best_ucb, best_action, best_child = ucb_values[0]
        return best_action, best_child

    def distribution_probability(self):
        """
        Return the distribution probability of choosing an action according
        to the number of visits of the children.
        """
        dist_probability = {}
        total_visits = sum(self.root.n_a.values())
        for action, visits in self.root.n_a.items():
            dist_probability[action] = visits/total_visits
        return dist_probability

    def add_dist_prob_to_children(self, node, dist_prob):
        """
        Add the probability distribution given from the network to 
        node's children. This probability is used later to calculate
        he selection phase of the UCT algorithm.
        """
        standard_dist = [((2,2),0), ((2,3),1), ((2,4),2), ((2,5),3), ((2,6),4),
                     ((3,2),5), ((3,3),6), ((3,4),7), ((3,5),8), ((3,6),9),
                     ((4,2),10), ((4,3),11), ((4,4),12), ((4,5),13), ((4,6),14),
                     ((5,2),15), ((5,3),16), ((5,4),17), ((5,5),18), ((5,6),19),
                     ((6,2),20), ((6,3),21), ((6,4),22), ((6,5),23), ((6,6),24),
                     ((2,),25), ((3,),26), ((4,),27), ((5,),28), ((6,),29),
                    ('y',30), ('n',31)]
        standard_dist = collections.OrderedDict(standard_dist)
        for key in node.children.keys():
            node.p_a[key] = dist_prob[standard_dist[key]]

    def valid_positions_channel(self, game_config):
        """
        Return a channel that fills with a value of 1 if that cell is valid and
        0 otherwise.
        """
        rows = game_config.column_range[1] - game_config.column_range[0] + 1
        columns = game_config.initial_height + game_config.offset * (rows//2)
        channel = np.zeros((rows, columns), dtype=int)
        height = game_config.initial_height
        for i in range(rows):
            for j in range(height):
                channel[i][j] = 1
            if i < rows//2:
                height += game_config.offset
            else:
                height -= game_config.offset
        return channel

    def finished_columns_channels(self, state, channel):
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

    def player_won_column_channels(self, state, channel):
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

    def player_turn_channel(self, state, channel):
        """
        Return a channel filled with 1 if it is the the player 1's turn and a channel
        filled with 0 otherwise.
        """
        shape = channel.shape
        if state.player_turn == 1:
            return np.ones(shape, dtype=int)
        else:
            return np.zeros(shape, dtype=int)

    def transform_to_input(self, game, game_config):
        """Receive the game state and return the six channels used as input for the network"""
        channel_valid = self.valid_positions_channel(game_config)
        channel_finished_1, channel_finished_2 = self.finished_columns_channels(game, channel_valid)
        channel_won_column_1, channel_won_column_2 = self.player_won_column_channels(game, channel_valid)
        channel_turn = self.player_turn_channel(game, channel_valid)
        list_of_channels = [channel_valid, channel_finished_1, channel_finished_2,
                            channel_won_column_1, channel_won_column_2, channel_turn]
        list_of_channels = np.array(list_of_channels)
        list_of_channels = np.expand_dims(list_of_channels, axis=0)
        list_of_channels = list_of_channels.reshape(list_of_channels.shape[0], 
                            list_of_channels.shape[2], list_of_channels.shape[3], -1)
        return list_of_channels

    def remove_invalid_actions(self, dist_prob, keys):
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

    def transform_dist_prob(self, dist_prob):
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

    def transform_actions_to_dist(self, actions):
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

    def transform_dataset_to_input(self, dataset_for_network):
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
        dist_probs_label = [self.transform_dist_prob(dist_dict) for dist_dict in dist_probs_label]
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