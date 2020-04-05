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
        """Expand the children of the "parent" node."""

        # parent might not have any children given the dice configuration.
        # Therefore, it should be checked if the player is busted. 
        # is_player_busted() automatically changes the game dynamics 
        # (player turn, etc) if the player is indeed busted.
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

    def rollout(self, node):
        """Take random actions until a game is finished and return the value."""

        # Special case where 'node' is a terminal state.
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

    def select_action(self, game, root, dist_probability):
        """Return the action with the highest visit score."""

        visit_counts = [(child.n_visits, action)
                      for action, child in root.children.items()]
        # Sort based on the number of visits
        visit_counts.sort(key=lambda t: t[0])
        _, action = visit_counts[-1]
        return action

    def calculate_ucb_max(self, node, action):
        """Return the node UCB1 value of the MAX player."""

        return node.q_a[action] + self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))

    def calculate_ucb_min(self, node, action):
        """Return the node UCB1 value of the MIN player."""
        
        return node.q_a[action] - self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))