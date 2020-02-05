from collections import defaultdict
from game import Game
import math, random
import numpy as np
import copy
from players.uct_player import UCTPlayer, Node

class Vanilla_UCT(UCTPlayer):

    def __init__(self, c, n_simulations):
        super().__init__(c, n_simulations)

    def expand_children(self, parent):
        valid_actions = parent.state.available_moves()
        if len(valid_actions) == 0:
            valid_actions = ['y', 'n']
        for action in valid_actions:
            child_game = parent.state.clone()
            child_game.play(action)
            child_state = Node(child_game, parent)
            child_state.action_taken = action
            parent.children[action] = child_state

    def rollout(self, node, scratch_game):
        """Take random actions until a game is finished and return the value."""
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

    def calculate_ucb_max(self, node, action):
        """
        Return the node UCB1 value of the MAX player.
        """
        return node.q_a[action] + self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))

    def calculate_ucb_min(self, node, action):
        """
        Return the node UCB1 value of the MIN player.
        """
        return node.q_a[action] - self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))