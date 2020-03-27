from collections import defaultdict
from game import Game
import math, random
import numpy as np
import copy
import collections
from players.alphazero_player import AlphaZeroPlayer, Node

class Network_UCT(AlphaZeroPlayer):

    def __init__(self, c, n_simulations, n_games, n_games_evaluate,
                    victory_rate, alphazero_iterations, column_range, 
                        offset, initial_height, dice_value, network):
        super().__init__(c, n_simulations, n_games, n_games_evaluate,
                            victory_rate, alphazero_iterations, column_range, 
                                offset, initial_height, dice_value, network)

    def rollout(self, node, game):
        """
        Retrieve the value (who would win the current simulation) from the network.
        """
        network_input = self.transform_to_input(game, self.column_range, self.offset, self.initial_height)
        valid_actions_dist = self.transform_actions_to_dist(node.state.available_moves())
        _, network_value_output = self.network.predict([network_input,valid_actions_dist], batch_size=1)
        return network_value_output[0][0]