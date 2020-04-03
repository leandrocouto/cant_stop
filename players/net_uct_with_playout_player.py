from collections import defaultdict
from game import Game
import math, random
import numpy as np
import copy
import collections
from players.alphazero_player import AlphaZeroPlayer, Node

class Network_UCT_With_Playout(AlphaZeroPlayer):

    def __init__(self, c, n_simulations, column_range, 
                        offset, initial_height, dice_value, network):
        super().__init__(c, n_simulations, column_range, 
                                offset, initial_height, dice_value, network)

    def rollout(self, node):
        """Take random actions until a game is finished and return the value."""
        
        # Special case where 'node' is a terminal state
        winner, terminal_state = node.state.is_finished()
        if terminal_state:
            if winner == 1:
                return 1
            else:
                return -1
            
        end_game = False
        game = node.state.clone()
        while not end_game:
            #avoid infinite loops in smaller boards
            who_won, end_game = game.is_finished()
            moves = game.available_moves()
            if game.is_player_busted(moves):
                continue
            chosen_move = random.choice(moves)
            game.play(chosen_move)
            who_won, end_game = game.is_finished()
        if who_won == 1:
            return 1
        else:
            return -1

    def clone(self, reg, conv_number):
        '''
        Clones self player.
        Tensorflow 1.3 complains about deepcopying keras models 
        while TF 2.0 > does not. Using this method in order to be able
        to run in either versions.
        Return a Network_UCT_With_Playout player.
        '''
        from main import define_model
        import pickle
        self_copy = Network_UCT_With_Playout(self.c, self.n_simulations, self.column_range, 
                        self.offset, self.initial_height, self.dice_value, None)
        self_copy.root = pickle.loads(pickle.dumps(self.root, -1))
        self_copy.action = pickle.loads(pickle.dumps(self.action, -1))
        self_copy.dist_probability = pickle.loads(pickle.dumps(self.dist_probability, -1))
        self_copy.network = define_model(reg, conv_number, self.column_range, 
                                        self.offset, self.initial_height, self.dice_value)
        self_copy.network.set_weights(self.network.get_weights())
        return self_copy