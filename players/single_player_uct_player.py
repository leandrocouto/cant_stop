import math, random
import numpy as np
import copy
import collections
from players.player import Player
from abc import abstractmethod
from collections import defaultdict
import time

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
    

class SinglePlayerUCTPlayer(Player):
    def __init__(self, c, n_simulations):
        """
        - root is a Node instance representing the root of the game tree.
        - action stores the action this player will return for the game passed
          as parameter in the get_action method.
        - dist_probability stores the distribution probability over all actions
          regarding the probability of choosing a certain action given the game
          passed as parameter in the get_action method.
        - q_a_root is the Q-value of the previous root (because at the end of
          one run the root changes automatically) over all possible actions.
        - c is the constant the balance exploration and exploitation.
        - n_simulations is the number of simulations made in the UCT algorithm.
        """
        self.root = None
        self.action = None
        self.dist_probability = None
        self.q_a_root = None
        self.c = c
        self.n_simulations = n_simulations

    def select_action(self, game, root, dist_probability):
        """ Return the final action after the UCT simulations. """
        
        visit_counts = [(child.n_visits, action)
                      for action, child in root.children.items()]
        # Sort based on the number of visits
        visit_counts.sort(key=lambda t: t[0])
        _, action = visit_counts[-1]
        return action

    def get_action(self, game):
        """ Return the action given by the UCT algorithm. """
        action, dist_probability, q_a_root = self.run_UCT(game)
        self.action = action
        self.dist_probability = dist_probability
        self.q_a_root = q_a_root
        return action

    def run_UCT(self, game):
        """Main routine of the UCT algoritm."""
        
        # If current tree is null, create one using game
        if self.root == None:
            self.root = Node(game.clone())
        # This means the player is still playing (i.e.: didn't choose 'n').
        else:
            # Therefore, check if current root has the same children as "game"
            # offers. If not, reset the tree.
            if set(self.root.children) != set(game.available_moves()):
                self.root = None
                self.root = Node(game.clone())

        #Expand the children of the root if it is not expanded already
        if not self.root.is_expanded():
            self.expand_children(self.root)

        root_state = self.root.state.clone()

        for _ in range(self.n_simulations):
            node = self.root
            node.state = root_state.clone()
            search_path = [node]
            while node.is_expanded():
                action, new_node = self.select_child(node)
                node.action_taken = action
                search_path.append(new_node)
                node = new_node
            # At this point, a leaf was reached.
            # If it was not visited yet, then perform the rollout and
            # backpropagates the reward returned from the simulation.
            # If it has been visited, then expand its children, choose the one
            # with the highest ucb score and do a rollout from there.
            if node.n_visits == 0:
                rollout_value = self.rollout(node)
                self.backpropagate(search_path, rollout_value)
            else:
                terminal_state = node.state.is_finished()
                # Special case: if "node" is actually a leaf of the game (not 
                # a leaf from the current tree), then only rollout should be 
                # applied since it does not make sense to expand the children
                # of a leaf.
                if terminal_state:
                    rollout_value = self.rollout(node)
                    self.backpropagate(search_path, rollout_value)
                else:
                    self.expand_children(node)
                    action, new_node = self.select_child(node)
                    node.action_taken = action
                    search_path.append(new_node)
                    node = new_node
                    rollout_value = self.rollout(node)
                    self.backpropagate(search_path, rollout_value)
        
        dist_probability = self.distribution_probability(game)
        action = self.select_action(game, self.root, dist_probability)
        q_a_root = self.root.q_a
        self.root = self.root.children[action]
        # Remove the statistics of the chosen action from the chosen child
        if self.root.n_a:
            self.root.n_a.pop(action)
        if self.root.q_a:
            self.root.q_a.pop(action)
        return action, dist_probability, q_a_root

    def get_tree_size(self, node):
        """
        Return the number of nodes in the tree starting from self.root.
        Currently works only when node == self.root.
        """

        # If the tree has not been created yet.
        if node == None:
            return 0
        n_nodes = 1
        for child in node.children:
            n_nodes += self.get_tree_size(node.children[child])
        return n_nodes

    def backpropagate(self, search_path, value):
        """Propagate the game value all the way up the tree to the root."""

        for node in search_path:
            node.n_visits += 1
            node.n_a[node.action_taken] += 1 
            # Incremental mean calculation
            node.q_a[node.action_taken] = (node.q_a[node.action_taken] * 
                                            (node.n_visits - 1) + value) / \
                                                node.n_visits

    def select_child(self, node):
        """Return the child Node with the highest UCB score."""
        ucb_values = []
        for action, child in node.children.items():
            if child.n_visits == 0:
                ucb_min = float('-inf')
            else:
                ucb_min =  self.calculate_ucb(node, action)
            ucb_values.append((ucb_min, action, child))
        # Sort the list based on the ucb score
        ucb_values.sort(key=lambda t: t[0])
        best_ucb, best_action, best_child = ucb_values[0]
        return best_action, best_child

    def expand_children(self, parent):
        """Expand the children of the "parent" node"""
        
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
        """ Return the game value of the end of the current simulation. """
        
        # Special case where 'node' is a terminal state.
        terminal_state = node.state.is_finished()
        if terminal_state:
            return 0

        end_game = False
        game = node.state.clone()
        rounds = 0
        while not end_game:
            #avoid infinite loops in smaller boards
            end_game = game.is_finished()
            moves = game.available_moves()
            if game.is_player_busted(moves):
                continue
            chosen_move = random.choice(moves)
            game.play(chosen_move)
            end_game = game.is_finished()
        return game.n_rounds

    def calculate_ucb(self, node, action):
        """ Return the node UCB value of the MIN player. """
        
        return node.q_a[action] - self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))

    def distribution_probability(self, game):
        """
        Return the distribution probability of choosing an action according
        to the number of visits of the children.
        """
        dist_probability = {}

        total_visits = sum(self.root.n_a.values())

        for action, visits in self.root.n_a.items():
            dist_probability[action] = visits/total_visits
        return dist_probability
