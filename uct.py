from collections import defaultdict
from game import Game
import math, random
import numpy as np

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
        #These will initialize value = 0 for whichever keys yet to be added.
        self.n_a = defaultdict(lambda: 0, self.n_a)
        self.q_a = defaultdict(lambda: 0, self.q_a)
        self.parent = parent
        self.children = {}
        self.action_taken = None

    def is_expanded(self):
        """Return a boolean"""
        return len(self.children) > 0

class MCTS:
    def __init__(self, c, n_simulations):
        self.c = c
        self.player = 0
        self.n_simulations = n_simulations

    def run_mcts(self, game):
        root = Node(game.clone())
        #Expands the children of the root before the actual algorithm
        self.expand_children(root)
        for _ in range(self.n_simulations):
            node = root
            scratch_game = game.clone()
            search_path = [node]

            while node.is_expanded():
                action, new_node = self.select_child(node)
                #print('nó selecionado')
                #new_node.state.board_game.print_board([])
                node.action_taken = action 
                scratch_game.play(action)
                search_path.append(new_node)
                node = new_node
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
            action = self.select_action(game, root)
        print('size of search path:', len(search_path))
        return action, root
    def expand_children(self, parent):
        valid_actions = parent.state.available_moves()
        for action in valid_actions:
            child_game = parent.state.clone()
            child_game.play(action)
            child_state = Node(child_game, parent)
            child_state.action_taken = action
            parent.children[action] = child_state

    def select_action(self, game, root):
        visit_counts = [(child.n_visits, action)
                      for action, child in root.children.items()]
        _, action = max(visit_counts)
        return action


    # Select the child with the highest UCB score.
    def select_child(self, node):
        _, action, child = max((self.ucb_value(node), action, child)
                             for action, child in node.children.items())
        return action, child

    def ucb_value(self, node):
        valid_actions = node.state.available_moves()
        #print('dentro de ucb. Node:')
        #node.state.board_game.print_board([])
        #print('n visits: ', node.n_visits)
        if node.n_visits == 0:
            return float('inf')
        else:
            if node.state.player_turn == 1:
                max_value_action = node.q_a[valid_actions[0]] + self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[valid_actions[0]]))
                #print('max value action:', max_value_action)
                for i in range(1, len(valid_actions)):
                    new_value = node.q_a[valid_actions[i]] + self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[valid_actions[i]]))
                    #print('new value ', i, ':', new_value)
                    if new_value > max_value_action:
                        max_value_action = new_value
                return max_value_action
            else:
                min_value_action = node.q_a[valid_actions[0]] - self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[valid_actions[0]]))
                for i in range(1, len(valid_actions)):
                    new_value = node.q_a[valid_actions[i]] - self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[valid_actions[i]]))
                    if new_value < min_value_action:
                        min_value_action = new_value
                return min_value_action


    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    def backpropagate(self, search_path, action, value):
        #print(  'backpropagate de tamanho do search path:', len(search_path))

        for node in search_path:
            # Ignore the last one
            if len(node.children) == 0:
                continue
            #node.state.board_game.print_board([])
            #print('action taken of this node: ', node.action_taken)
            node.n_visits += 1
            node.n_a[node.action_taken] += 1 
            node.q_a[node.action_taken] = (node.q_a[node.action_taken] * (node.n_visits - 1) 
                                                                + value) / node.n_visits 
    def rollout(self, node, scratch_game):
        end_game = False
        #print('antes rollout')
        #scratch_game.board_game.print_board(scratch_game.player_won_column)
        while not end_game:
            moves = scratch_game.available_moves()
            if scratch_game.is_player_busted(moves):
                continue
            chosen_move = random.choice(moves)
            scratch_game.play(chosen_move)
            #scratch_game.board_game.print_board(scratch_game.player_won_column)
            who_won, end_game = scratch_game.is_finished()
        if who_won == 1:
            return 1
        else:
            return -1

