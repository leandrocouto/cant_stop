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
        - p_a is a dictionary {key, value} where key is the action taken from
          this node and value is the probability given by the trained network
          to choose this action. Value updated in the expand_children() method
          from the MCTS class.
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
        self.p_a = {}
        # These will initialize value = 0 for whichever keys yet to be added.
        self.n_a = defaultdict(lambda: 0, self.n_a)
        self.q_a = defaultdict(lambda: 0, self.q_a)
        self.p_a = defaultdict(lambda: 0, self.q_a)
        self.parent = parent
        self.children = {}
        self.action_taken = None


    def is_expanded(self):
        """Return a boolean."""
        return len(self.children) > 0
    

class UCTPlayer(Player):
    @abstractmethod
    def __init__(self, c, n_simulations):
        """
        - root is a Node instance representing the root of the game tree.
        - action stores the action this player will return for the game passed
          as parameter in the get_action method.
        - dist_probability stores the distribution probability over all actions
          regarding the probability of choosing a certain action given the game
          passed as parameter in the get_action method.
        - c is the constant the balance exploration and exploitation.
        - n_simulations is the number of simulations made in the UCT algorithm.
        """
        self.root = None
        self.action = None
        self.dist_probability = None
        self.c = c
        self.n_simulations = n_simulations


    @abstractmethod
    def expand_children(self, parent):
        """Expand the children of the "parent" node"""
        pass

    @abstractmethod
    def rollout(self, node, scratch_game):
        """
        Return the game value of the end of the current simulation.
        Concrete classes must implement this method.
        """
        pass

    @abstractmethod
    def calculate_ucb_max(self, node):
        """
        Return the node UCB value of the MAX player.
        Concrete classes must implement this method.
        """
        pass

    @abstractmethod
    def calculate_ucb_min(self, node):
        """
        Return the node UCB value of the MIN player.
        Concrete classes must implement this method.
        """
        pass

    @abstractmethod
    def select_action(self):
        """
        Return the final action after the UCT simulations.
        Concrete classes must implement this method.
        """
        pass

    def reset_tree(self):
        """
        Reset the tree. In the current implementation, UCT keeps the
        relevant tree for future UCT simulations. Use this method if you
        will play several games in a row using the same UCTPlayer.
        If this is not called, the tree will grow indefinitely after many games,
        resulting in a stack overflow.
        """
        self.root = None
        self.action = None
        self.dist_probability = None

    def get_action(self, game, actions_taken):
        """ Return the action given by the UCT algorithm. """
        action, dist_probability = self.run_UCT(game, actions_taken)
        self.action = action
        self.dist_probability = dist_probability
        return action

    def get_dist_probability(self):
        """ 
        Return the actions distribution probability regarding
        the game passed as parameter in get_action. Should be 
        called after get_action.
        """
        return self.dist_probability

    def run_UCT(self, game, actions_taken):
        """Main routine of the UCT algoritm."""

        # If current tree is null, create one using game
        if self.root == None:
            self.root = Node(game.clone())
        # If it is not, check if there are actions from actions_taken to update the tree from this player.
        # If the list is not empty, update the root accordingly.
        # If the list is empty, that means this player hasn't passed the turn, therefore the root is 
        # already updated.
        elif len(actions_taken) != 0:
            for i in range(len(actions_taken)):
                # Check if action from history in in current self.root.children
                if actions_taken[i][0] in self.root.children:
                    # Check if the actions are made from the same player
                    if self.root.children[actions_taken[i][0]].state.player_turn == actions_taken[i][1] and \
                        set(self.root.state.available_moves()) == set(game.available_moves()):
                        self.root = self.root.children[actions_taken[i][0]]
                    else:
                        self.root = None
                        self.root = Node(actions_taken[i][2].clone())
                else:
                    self.root = None
                    self.root = Node(actions_taken[i][2].clone())
        # This means the player is still playing (i.e.: didn't choose 'n' action)
        else:
            # Therefore, check if current root has the same children as "game" offers.
            # If not, reset the tree.
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
            #At this point, a leaf was reached.
            #If it was not visited yet, then perform the rollout and
            #backpropagates the reward returned from the end of the simulation.
            #If it has been visited, then expand its children, choose the one
            #with the highest ucb score and do a rollout from there.
            if node.n_visits == 0:
                rollout_value = self.rollout(node)
                self.backpropagate(search_path, rollout_value)
            else:
                _, terminal_state = node.state.is_finished()
                # Special case: if "node" is actually a leaf of the game (not a leaf
                # from the current tree), then only rollout should be applied since
                # it does not make sense to expand the children of a leaf.
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
        self.root = self.root.children[action]
        if self.root.n_a:
            self.root.n_a.pop(action)

        return action, dist_probability 

    def get_tree_size(self, node):
        """
        Return the number of nodes in the tree starting from self.root.
        Currently works only when node == self.root.
        """

        # If the tree has not been created yet 
        if node == None:
            return 0
        n_nodes = 1
        for child in node.children:
            n_nodes += self.get_tree_size(node.children[child])
        return n_nodes

    def backpropagate(self, search_path, value):
        """Propagate the value from rollout all the way up the tree to the root."""
        
        for node in search_path:
            node.n_visits += 1
            node.n_a[node.action_taken] += 1 
            # Incremental mean calculation
            node.q_a[node.action_taken] = (node.q_a[node.action_taken] * 
                                            (node.n_visits - 1) + value) / \
                                                node.n_visits
        #exit()

    def select_child(self, node):
        """Return the child Node with the highest UCB score."""
        ucb_values = []
        for action, child in node.children.items():
            if node.state.player_turn == 1:
                if child.n_visits == 0:
                    ucb_max = float('inf')
                else:
                    ucb_max =  self.calculate_ucb_max(node, action)
                ucb_values.append((ucb_max, action, child))
            else:
                if child.n_visits == 0:
                    ucb_min = float('-inf')
                else:
                    ucb_min =  self.calculate_ucb_min(node, action)
                ucb_values.append((ucb_min, action, child))
        # Sort the list based on the ucb score
        ucb_values.sort(key=lambda t: t[0])
        if node.state.player_turn == 1:
            best_ucb, best_action, best_child = ucb_values[-1]
        else:
            best_ucb, best_action, best_child = ucb_values[0]
        return best_action, best_child

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