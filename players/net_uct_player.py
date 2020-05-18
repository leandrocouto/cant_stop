from collections import defaultdict
from game import Game
import math, random
import numpy as np
import copy
import collections
from players.alphazero_player import AlphaZeroPlayer, Node
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

    def clone(self):
        """
        Clone self player except for the network (due to tf.keras issues when
        using tf.keras models in parallel calls).
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
        self_copy.network = None
        return self_copy