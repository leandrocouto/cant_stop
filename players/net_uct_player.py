from collections import defaultdict
from game import Game
from utils import transform_to_input, remove_invalid_actions
from utils import transform_actions_to_dist
import math, random
import numpy as np
import copy
import collections
from players.alphazero_player import AlphaZeroPlayer, Node

class Network_UCT(AlphaZeroPlayer):

    def __init__(self, alphazero_config, game_config, network):
        super().__init__(alphazero_config)
        self.game_config = game_config
        self.network = network
    
    def expand_children(self, parent):
        """Expand the children of the "parent" node"""
        valid_actions = parent.state.available_moves()
        if len(valid_actions) == 0:
            valid_actions = ['y', 'n']
        for action in valid_actions:
            child_game = parent.state.clone()
            child_game.play(action)
            child_state = Node(child_game, parent)
            child_state.action_taken = action
            parent.children[action] = child_state

        #Update the  distribution probability of the children (node.p_a)
        network_input_parent = transform_to_input(parent.state, self.game_config)
        valid_actions_dist = transform_actions_to_dist(valid_actions)
        dist_prob, _= self.network.predict([network_input_parent, valid_actions_dist], batch_size=1)
        dist_prob = remove_invalid_actions(dist_prob[0], parent.children.keys())
        self.add_dist_prob_to_children(parent, dist_prob)

    def rollout(self, node, scratch_game):
        """
        Retrieve the value (who would win the current simulation) from the network.
        """
        network_input = transform_to_input(scratch_game, self.game_config)
        valid_actions_dist = transform_actions_to_dist(node.state.available_moves())
        _, network_value_output = self.network.predict([network_input,valid_actions_dist], batch_size=1)
        return network_value_output[0][0]

    def calculate_ucb_max(self, node, action):
        """
        Return the node modified PUCT (see AZ paper) value of the MAX player.
        """
        return node.q_a[action] + self.alphazero_config.c * node.p_a[action] * np.divide(math.sqrt(math.log(node.n_visits)), node.n_a[action])

    def calculate_ucb_min(self, node, action):
        """
        Return the node modified PUCT (see AZ paper) value of the MIN player.
        """
        return node.q_a[action] - self.alphazero_config.c * node.p_a[action] * np.divide(math.sqrt(math.log(node.n_visits)), node.n_a[action])

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
