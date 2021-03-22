from collections import defaultdict
from game import Game
import math, random
import numpy as np
import copy
from players.player import Player
from MetropolisHastings.DSL import DSL
from players.uct_player import UCTPlayer, Node

# Forte - 10k iterações
'''
class RolloutScript(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = 34 + DSL.number_cells_advanced_this_round(state) 
            if score_yes_no < 39 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 12 , 1 , 12 , 12 , 6 , 13 , 18 , 3 , 9 , 18 , 5 ] 
            for i in range(len(actions)): 
                score_columns[i] = 7 + DSL.advance_in_action_col(actions[i]) + DSL.get_weight_for_action_columns(actions[i], weights) - abs( DSL.does_action_place_new_neutral(actions[i], state) * 35 ) 
            return actions[np.argmax(score_columns)] 
'''
'''
# Razoavel - Iteração 5k
class RolloutScript(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = DSL.number_cells_advanced_this_round(state) 
            if score_yes_no < 5 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 14 , 1 , 14 , 2 , 18 , 4 , 8 , 2 , 16 , 4 , 18 ] 
            for i in range(len(actions)): 
                score_columns[i] = abs( DSL.get_weight_for_action_columns(actions[i], weights) * abs( DSL.get_weight_for_action_columns(actions[i], weights) + abs( 5 - 22 ) - 13 ) ) 
            return actions[np.argmax(score_columns)] 
'''
# Fraco - Iteração 1k
class RolloutScript(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = DSL.number_cells_advanced_this_round(state) 
            if score_yes_no < 5 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 11 , 4 , 3 , 10 , 16 , 8 , 14 , 3 , 3 , 12 , 12 ] 
            for i in range(len(actions)): 
                score_columns[i] = DSL.get_weight_for_action_columns(actions[i], weights) - DSL.does_action_place_new_neutral(actions[i], state) * abs( DSL.does_action_place_new_neutral(actions[i], state) + DSL.does_action_place_new_neutral(actions[i], state) ) 
            return actions[np.argmax(score_columns)] 

class Boosted_Vanilla_UCT(UCTPlayer):

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

        rollout_script = RolloutScript()
        while not end_game:
            #avoid infinite loops in smaller boards
            who_won, end_game = game.is_finished()
            moves = game.available_moves()
            if game.is_player_busted(moves):
                continue
            chosen_move = rollout_script.get_action(game)
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