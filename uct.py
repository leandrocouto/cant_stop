from collections import defaultdict
from game import Game
import math

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
        - children_list is a dictionary {key, value} where key is the action 
          taken from this node and value is the resulting node after applying 
          the action.
        """
        self.state = state
        self.n_visits = 0
        self.n_a = {}
        self.q_a = {}
        #These will initialize value = 0 for whichever keys yet to be added.
        self.n_a = defaultdict(lambda: 0, self.n_a)
        self.q_a = defaultdict(lambda: 0, self.q_a)
        self.parent = parent
        self.children_list = {}

    def is_expanded(self):
        """Return a boolean"""
        return len(self.children) > 0

def run_mcts(game, n_simulations):
    root = Node(game.board)

    for _ in range(n_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.is_expanded():
            action, node = select_child(node)
            scratch_game.apply(action)
            search_path.append(node)
        #At this point, a leaf was reached.
        #If it was not visited yet, then perform the rollout and backpropagates
        #the reward returned from the end of the simulation
        if node.n_visits == 0:
            rollout_value = rollout(node, scratch_game)
            backpropagate(search_path, rollout_value, scratch_game.to_play())
        else:

        

        #TODO: expand the node



        action = select_action(game, root)
    return action, root


def select_action(game, root):
    visit_counts = [(child.visit_count, action)
                  for action, child in root.children.iteritems()]
    _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.
def select_child(node):
    _, action, child = max((ucb_score(node, child), action, child)
                         for action, child in node.children.iteritems())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_value(node, player_turn):
    valid_actions = game.valid_actions(node.state)
    if node.n_visits == 0:
        return float('inf')
    else:
        if player_turn == 'MAX':
            max_value_action = valid_actions[0] 
            for action in valid_actions:
                new_value = node.q_a[action] + c * math.sqrt(
                            math.log(node.n_visits) / node.n_a[action])
                if 


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path, value, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1

def rollout(node, scratch_game):
    return 1

