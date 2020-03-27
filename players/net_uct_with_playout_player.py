from collections import defaultdict
from game import Game
import math, random
import numpy as np
import copy
import collections
from players.alphazero_player import AlphaZeroPlayer, Node

class Network_UCT_With_Playout(AlphaZeroPlayer):

    def __init__(self, c, n_simulations, n_games, n_games_evaluate,
                    victory_rate, alphazero_iterations, column_range, 
                        offset, initial_height, dice_value, network):
        super().__init__(c, n_simulations, n_games, n_games_evaluate,
                            victory_rate, alphazero_iterations, column_range, 
                                offset, initial_height, dice_value, network)

    def rollout(self, node, scratch_game):
        """Take random actions until a game is finished and return the value."""
        # Special case where 'node' is a terminal state
        winner, terminal_state = node.state.is_finished()
        if terminal_state:
            return winner
            
        end_game = False
        while not end_game:
            #avoid infinite loops in smaller boards
            who_won, end_game = scratch_game.is_finished()
            moves = scratch_game.available_moves()
            if scratch_game.is_player_busted(moves):
                continue
            chosen_move = random.choice(moves)
            scratch_game.play(chosen_move)
            who_won, end_game = scratch_game.is_finished()
        if who_won == 1:
            return 1
        else:
            return -1