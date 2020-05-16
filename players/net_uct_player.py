from collections import defaultdict
from game import Game
import math, random
import numpy as np
import copy
import collections
from players.alphazero_player import AlphaZeroPlayer, Node
from models import define_model
import pickle


class Network_UCT(AlphaZeroPlayer):

    def __init__(self, c, n_simulations, column_range, 
                        offset, initial_height, dice_value, network):
        super().__init__(c, n_simulations, column_range, 
                                offset, initial_height, dice_value, network)

    def rollout(self, node):
        """
        Retrieve the value (who would win the current simulation) 
        from the network.
        """

        network_input = self.transform_to_input(node.state, self.column_range,
                                                self.offset, 
                                                self.initial_height
                                                )
        valid_actions_dist = self.transform_actions_to_dist(
                                            node.state.available_moves()
                                            )

        _, network_value_output = self.network.predict(
                                    [network_input,valid_actions_dist]
                                    )
        return network_value_output[0][0]

    def clone(self, reg, conv_number):
        """
        Clone self player.
        Tensorflow 1.3 complains about deepcopying keras models 
        while TF 2.0 > does not. Using this method in order to be able
        to run in either versions.
        Return a Network_UCT player.
        """
        self_copy = Network_UCT(self.c, self.n_simulations, self.column_range, 
                        self.offset, self.initial_height, 
                        self.dice_value, None)
        self_copy.root = pickle.loads(pickle.dumps(self.root, -1))
        self_copy.action = pickle.loads(pickle.dumps(self.action, -1))
        self_copy.dist_probability = pickle.loads(
                                    pickle.dumps(self.dist_probability, -1)
                                    )
        self_copy.network = define_model(reg, conv_number, self.column_range, 
                                        self.offset, self.initial_height, 
                                        self.dice_value
                                        )
        self_copy.network.set_weights(self.network.get_weights())
        #self_copy = pickle.loads(pickle.dumps(self, -1))
        return self_copy