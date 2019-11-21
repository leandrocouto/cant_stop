from collections import defaultdict
from game import Game
from utils import transform_to_input, remove_invalid_actions
from utils import transform_actions_to_dist
import math, random
import numpy as np
import copy
import collections

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


class MCTS:
    def __init__(self, config, network):
        self.config = config
        self.root = None
        self.network = network


    def run_mcts(self, game):
        """Main routine of the MCTS algoritm."""
        if self.root == None:
            self.root = Node(game.clone())
        else:
            self.root.state = game.clone()
        #Expand the children of the root before the actual algorithm
        self.expand_children(self.root)

        #print('dentro run antes for')
        for _ in range(self.config.n_simulations):
            node = self.root
            scratch_game = game.clone()
            search_path = [node]
            #print('antes while')
            while node.is_expanded():
                action, new_node = self.select_child(node)
                node.action_taken = action 
                scratch_game.play(action)
                search_path.append(new_node)
                node = copy.deepcopy(new_node)
            #print('dps while')
            #At this point, a leaf was reached.
            #If it was not visited yet, then get the calculated rollout value
            #from the network and backpropagates the reward returned from the
            #end of the simulation.
            #If it has been visited, then expand its children, choose the one
            #with the highest ucb score and calculated rollout value from the
            #network.
            if node.n_visits == 0:
                network_input = transform_to_input(scratch_game, self.config)
                valid_actions_dist = transform_actions_to_dist(node.state.available_moves())
                _, network_value_output = self.network.predict([network_input,valid_actions_dist], batch_size=1)
                rollout_value = network_value_output[0][0]
                self.backpropagate(search_path, action, rollout_value)
            else:
                self.expand_children(node)
                action_for_rollout, node_for_rollout = self.select_child(node)
                search_path.append(node)
                network_input = transform_to_input(scratch_game, self.config)
                valid_actions_dist = transform_actions_to_dist(node.state.available_moves())
                _, network_value_output = self.network.predict([network_input,valid_actions_dist], batch_size=1)
                rollout_value = network_value_output[0][0]
                self.backpropagate(search_path, action_for_rollout, rollout_value)
        action = self.select_action(game, self.root)
        dist_probability = self.distribution_probability()
        self.root = self.root.children[action]
        return action, dist_probability


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

        #Update the  distribution probability of the children (node.p_a)
        network_input_parent = transform_to_input(parent.state, self.config)
        valid_actions_dist = transform_actions_to_dist(valid_actions)
        dist_prob, _= self.network.predict([network_input_parent, valid_actions_dist], batch_size=1)
        dist_prob = remove_invalid_actions(dist_prob[0], parent.children.keys())
        self.add_dist_prob_to_children(parent, dist_prob)


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
                    ucb_max =  node.q_a[action] + self.config.c * node.p_a[action] * np.divide(math.sqrt(math.log(node.n_visits)), node.n_a[action])

                ucb_values.append((ucb_max, action, child))
            else:
                if child.n_visits == 0:
                    ucb_min = float('-inf')
                else:
                    ucb_min =  node.q_a[action] - self.config.c * node.p_a[action] * np.divide(math.sqrt(math.log(node.n_visits)), node.n_a[action])
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

    def distribution_probability(self):
        """
        Return the distribution probability of choosing an action according
        to the number of visits of the children.
        """
        dist_probability = {}
        total_visits = sum(self.root.n_a.values())
        for action, visits in self.root.n_a.items():
            dist_probability[action] = visits/total_visits
        return dist_probability

    def add_dist_prob_to_children(self, node, dist_prob):
        """
        Add the probability distribution given from the network to 
        node's children. This probability is used later to calculate
        he selection phase of the UCT algorithm.
        """
        standard_dist = [((2,2),0), ((2,3),1), ((2,4),2), ((2,5),3), ((2,6),4),
                     ((3,2),5), ((3,3),6), ((3,4),7), ((3,5),8), ((3,6),9),
                     ((4,2),10), ((4,3),11), ((4,4),12), ((4,5),13), ((4,6),14),
                     ((5,2),15), ((5,3),16), ((5,4),17), ((5,5),18), ((5,6),19),
                     ((6,2),20), ((6,3),21), ((6,4),22), ((6,5),23), ((6,6),24),
                     ((2,),25), ((3,),26), ((4,),27), ((5,),28), ((6,),29),
                    ('y',30), ('n',31)]
        standard_dist = collections.OrderedDict(standard_dist)
        for key in node.children.keys():
            node.p_a[key] = dist_prob[standard_dist[key]]
