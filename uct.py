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
        self.root = None

    def run_mcts(self, game):
        if self.root == None:
            self.root = Node(game.clone())
        else:
            self.root.state = game.clone()

        #print('n visits do root: ',self.root.n_visits)    
        #print('size of root children: ', len(self.root.children))
        #print(self.root.children)
        #print('PRINTING ALL ROOT CHILDREN')
        #for key in self.root.children:
        #    self.root.children[key].state.board_game.print_board([])
        #    print()
        #    print()
        #Expands the children of the root before the actual algorithm
        self.expand_children(self.root)
        for _ in range(self.n_simulations):
            node = self.root
            scratch_game = game.clone()
            search_path = [node]
            #print('NODE ANTES - n visits: ', node.n_visits)
            #   node.state.board_game.print_board([])
            while node.is_expanded():
                #print('A')
                action, new_node = self.select_child(node)
                #print('nó selecionado')
                #new_node.state.board_game.print_board([])
                node.action_taken = action 
                scratch_game.play(action)
                search_path.append(new_node)
                node = copy.deepcopy(new_node)
            #print('NODE DEPOIS - n visits: ', node.n_visits)
            #node.state.board_game.print_board([])
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
                #print('B')
                action_for_rollout, node_for_rollout = self.select_child(node)
                search_path.append(node)

                rollout_value = self.rollout(node_for_rollout, scratch_game)
                self.backpropagate(search_path, action_for_rollout, rollout_value)
        action = self.select_action(game, self.root)
        #print('size of search path:', len(search_path))
        self.root = self.root.children[action]
        return action, self.root
    def expand_children(self, parent):
        valid_actions = parent.state.available_moves()
        if len(valid_actions) == 0:
            #print('\n\n\n\n\nTALVEZ FOI BUSTED???\n\n\n\n\n')
            valid_actions = ['y', 'n']
        #print('parent')
        #parent.state.board_game.print_board([])
        #print('EXPAND CHILDREN, valid actions: ', valid_actions)
        #print('EXPAND CHILDREN, NUMBER OF CHILDREN:', len(parent.children))
        for action in valid_actions:
            child_game = parent.state.clone()
            child_game.play(action)
            child_state = Node(child_game, parent)
            child_state.action_taken = action
            parent.children[action] = child_state
        #print('FIM EXPAND CHILDREN, NUMBER OF CHILDREN:', len(parent.children))

    def select_action(self, game, root):
        visit_counts = [(child.n_visits, action)
                      for action, child in root.children.items()]
        #print(visit_counts)
        visit_counts.sort(key=lambda t: t[0])
        #print('(n_visits, action): ', visit_counts)
        _, action = visit_counts[-1]#max(visit_counts)
        return action


    # Select the child with the highest UCB score.
    def select_child(self, node):
        ucb_values = []
        for action, child in node.children.items():
            if node.state.player_turn == 1:
                if child.n_visits == 0:
                    ucb_max = float('inf')
                else:
                    ucb_max =  node.q_a[action] + self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))

                ucb_values.append((ucb_max, action, child))
            else:
                if child.n_visits == 0:
                    ucb_min = float('-inf')
                else:
                    ucb_min =  node.q_a[action] - self.c * math.sqrt(
                                    np.divide(math.log(node.n_visits),
                                  node.n_a[action]))
                ucb_values.append((ucb_min, action, child))
        #for values in ucb_values:
        #    if values[1] in ['y', 'n'] and values[0] != float('inf') and values[0] != float('-inf'):
        #        print('jogador', node.state.player_turn)
        #        print('ucb: ', values[0])
        #        print('action: ', values[1])
        #        print('child')
        #        child.state.board_game.print_board([])
        ucb_values.sort(key=lambda t: t[0])
        if len(ucb_values) == 0:
            print('LISTA VAZIA')
            node.state.board_game.print_board([])
            print('len children do node acima: ', len(node.children))
        if node.state.player_turn == 1:
            best_ucb, best_action, best_child = ucb_values[-1]#max(ucb_values)
        else:
            best_ucb, best_action, best_child = ucb_values[0]#min(ucb_values)
        return best_action, best_child

    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    def backpropagate(self, search_path, action, value):
        for node in search_path:
            # Ignore the last one
            #if len(node.children) == 0:
            #    continue
            node.n_visits += 1
            node.n_a[node.action_taken] += 1 
            node.q_a[node.action_taken] = (node.q_a[node.action_taken] * 
                                            (node.n_visits - 1) + value) / \
                                                node.n_visits 
    def rollout(self, node, scratch_game):
        end_game = False
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

