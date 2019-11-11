from collections import defaultdict
from game import Game
import math, random
import numpy as np
import copy

class Node:
    def __init__(self, state, parent=None):
        """
        - n_visits is the number of visits in this node
        - n_a is a dictionary {key, value} where key is the action taken from 
          this node and value is the number of times this action was chosen.
        - q_a is a dictionary {key, value} where key is the action taken from
          this node and value is the mean reward of the simulation that passed
          through this node and used action 'a'.
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
        # These will initialize value = 0 for whichever keys yet to be added.
        self.n_a = defaultdict(lambda: 0, self.n_a)
        self.q_a = defaultdict(lambda: 0, self.q_a)
        self.parent = parent
        self.children = {}
        self.action_taken = None


    def is_expanded(self):
        """Return a boolean."""
        return len(self.children) > 0


class MCTS:
    def __init__(self, config):
        self.config = config
        self.root = None


    def run_mcts(self, game):
        """Main routine of the MCTS algoritm."""
        if self.root == None:
            self.root = Node(game.clone())
        else:
            self.root.state = game.clone()
        #Expands the children of the root before the actual algorithm
        self.expand_children(self.root)

        for _ in range(self.config.n_simulations):
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
        self.root = self.root.children[action]
        return action, self.root


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


    def select_action(self, game, root):
        """Return the action with the highest visit score."""
        visit_counts = [(child.n_visits, action)
                      for action, child in root.children.items()]
        # Sort based on the number of visits
        visit_counts.sort(key=lambda t: t[0])
        _, action = visit_counts[-1]
        return action


    def select_child(self, node):
        """Return the child Node with the highest UCB score."""
        ucb_values = []
        for action, child in node.children.items():
            if node.state.player_turn == 1:
                if child.n_visits == 0:
                    ucb_max = float('inf')
                else:
                    ucb_max =  node.q_a[action] + self.config.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))

                ucb_values.append((ucb_max, action, child))
            else:
                if child.n_visits == 0:
                    ucb_min = float('-inf')
                else:
                    ucb_min =  node.q_a[action] - self.config.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))
                ucb_values.append((ucb_min, action, child))
        # Sort the list based on the ucb score
        ucb_values.sort(key=lambda t: t[0])
        if node.state.player_turn == 1:
            best_ucb, best_action, best_child = ucb_values[-1]
        else:
            best_ucb, best_action, best_child = ucb_values[0]
        return best_action, best_child


    def backpropagate(self, search_path, action, value):
        """Propagate the value from rollout all the way up the tree to the root."""
        for node in search_path:
            node.n_visits += 1
            node.n_a[node.action_taken] += 1 
            # Incremental mean calculation
            node.q_a[node.action_taken] = (node.q_a[node.action_taken] * 
                                            (node.n_visits - 1) + value) / \
                                                node.n_visits


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
