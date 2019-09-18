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
        - action_taken is the action taken from parent to get to this node.
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
        self.action_taken = None

    def is_expanded(self):
        """Return a boolean"""
        return len(self.children) > 0

class MCTS:
    def __init__(self, c, n_simulations):
        self.c = c
        self.player = 0
        self.n_simulations = n_simulations

    def run_mcts(game):
        root = Node(game.board)
        #Expands the children of the root before the actual algorithm
        valid_actions = game.valid_actions(game.board)
        for action in valid_actions:
            child_state = Node(game.apply_action(action), root)
            child_state.action_taken = action
            root.children_list.append(child_state)
        for _ in range(self.n_simulations):
            node = root
            scratch_game = game.clone()
            search_path = [node]

            while node.is_expanded():
                action, node = select_child(node)
                scratch_game.apply(action)
                search_path.append(node)
            #At this point, a leaf was reached.
            #If it was not visited yet, then perform the rollout and
            #backpropagates the reward returned from the end of the simulation
            #If it has been visited, then expand its children, choose the one
            #with the highest ucb score and do a rollout from there.
            if node.n_visits == 0:
                rollout_value = rollout(node, scratch_game)
                backpropagate(search_path, rollout_value, scratch_game.to_play())
            else:
                valid_actions = game.valid_actions(node.state.board)

            

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

    def ucb_value(node):
        valid_actions = game.valid_actions(node.state)
        if node.n_visits == 0:
            return float('inf')
        else:
            if self.player == 0:
                max_value_action = valid_actions[0] 
                for i in range(1, len(valid_actions)):
                    new_value = node.q_a[valid_actions[i]] + self.c * math.sqrt(
                                math.log(node.n_visits)
                                 / node.n_a[valid_actions[i]])
                    if new_value > max_value_action:
                        max_value_action = valid_actions[i]
                return max_value_action
            else:
                min_value_action = valid_actions[0] 
                for i in range(1, len(valid_actions)):
                    new_value = node.q_a[valid_actions[i]] - self.c * math.sqrt(
                                math.log(node.n_visits)
                                 / node.n_a[valid_actions[i]])
                    if new_value < min_value_action:
                        min_value_action = valid_actions[i]
                return min_value_action


    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    def backpropagate(search_path, value, to_play):
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else (1 - value)
            node.visit_count += 1

    def rollout(node, scratch_game):
        return 1

