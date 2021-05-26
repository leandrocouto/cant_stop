import math
import sys
import pickle
import time
import os
import copy
import random
import matplotlib.pyplot as plt
from DSL import *
from rule_of_28_sketch import Rule_of_28_Player_PS
from rule_of_28_sketch import Rule_of_28_Player_PS_New
sys.path.insert(0,'..')
from game import Game
from play_game_template import simplified_play_single_game

class SimulatedAnnealing:
    def __init__(self, n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name, initial_time, max_time):
        self.n_SA_iterations = n_SA_iterations
        self.max_game_rounds = max_game_rounds
        self.n_games = n_games
        self.init_temp = init_temp
        self.d = d
        self.curr_id = 0
        self.time_elapsed = []
        self.wins_vs_glenn_x = []
        self.wins_vs_glenn_y = []
        self.algo_name = algo_name
        self.initial_time = initial_time
        self.max_time = max_time
        self.folder = algo_name + '_SA' + str(self.max_time) + '/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.log_file = self.folder + 'log.txt'
    
    def get_object(self, program_1, program_2):
        #return Rule_of_28_Player_PS(program_1, program_2)
        return Rule_of_28_Player_PS_New(program_1, program_2)

    def get_glenn_player(self):
        program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
        program_decide_column = Argmax(Map(Function(Sum(Map(Function(Minus(Times(NumberAdvancedByAction(), VarScalarFromArray('move_value')), Times(VarScalar('marker'), IsNewNeutral()))), NoneNode()))), VarList('actions')))
        return Rule_of_28_Player_PS(program_yes_no, program_decide_column)

    def id_tree_nodes(self, node):

        node.id = self.curr_id
        self.curr_id += 1
        for child in node.children:
            self.id_tree_nodes(child)

    def find_node(self, program, index):
        """ Return the tree node with the corresponding id. """

        return self._find_node(program, index)

    def _find_node(self, program, index):
        """ Helper method for self.find_node. """

        if program.id == index:
            return program
        else:
            for child_node in program.children:
                found_node = self._find_node(child_node, index)
                if found_node:
                    return found_node
        return None

    def get_traversal(self, program):
        list_of_nodes = []
        self._get_traversal(program, list_of_nodes)
        return list_of_nodes

    def _get_traversal(self, program, list_of_nodes):
        list_of_nodes.append(program)
        for child in program.children:
            self._get_traversal(child, list_of_nodes)


    def update_parent(self, program, parent):
        program.add_parent(parent)
        for child in program.children:
            self.update_parent(child, program)

    def update_children(self, program):
        children = program.children
        if children:
            program.add_children(children)
        for child in children:
            self.update_children(child)

    def finish_tree(self, node, chosen_node):
        """
        chosen_node is the DSL term (a string) that will substitute node. The 
        remaining children will be built recursively.
        """
        parent = node.parent
        if chosen_node == 'VarScalar':
            acceptable_nodes = [VarScalar('marker')]
            new_node = random.choice(acceptable_nodes)
            return new_node
        elif chosen_node == 'VarScalarFromArray':
            acceptable_nodes = ['progress_value', 'move_value']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'progress_value':
                new_node = VarScalarFromArray('progress_value')
            else:
                new_node = VarScalarFromArray('move_value')
            return new_node
        elif chosen_node == 'VarList':
            acceptable_nodes = ['actions', 'neutrals', 'None']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'actions':
                new_node =  VarList('actions')
            elif chosen == 'neutrals':
                new_node = VarList('neutrals')
            else:
                new_node = NoneNode()
            return new_node
        elif chosen_node == 'Constant':
            acceptable_nodes = [i for i in range(20)]
            chosen = random.choice(acceptable_nodes)
            return Constant(chosen)
        elif chosen_node == 'functions_scalars':
            acceptable_nodes = ['NumberAdvancedThisRound', 'NumberAdvancedByAction', 'IsNewNeutral', 'WillPlayerWinAfterN', 'AreThereAvailableColumnsToPlay', 'PlayerColumnAdvance', 'OpponentColumnAdvance']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'NumberAdvancedThisRound':
                new_node = NumberAdvancedThisRound()
            elif chosen == 'NumberAdvancedByAction':
                new_node = NumberAdvancedByAction()
            elif chosen == 'IsNewNeutral':
                new_node = IsNewNeutral()
            elif chosen == 'WillPlayerWinAfterN':
                new_node = WillPlayerWinAfterN()
            elif chosen == 'AreThereAvailableColumnsToPlay':
                new_node = AreThereAvailableColumnsToPlay()
            elif chosen == 'PlayerColumnAdvance':
                new_node = PlayerColumnAdvance()
            elif chosen == 'OpponentColumnAdvance':
                new_node = OpponentColumnAdvance()
            return new_node
        elif chosen_node == 'boolean_function':
            acceptable_nodes = ['IsNewNeutral', 'WillPlayerWinAfterN', 'AreThereAvailableColumnsToPlay']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'IsNewNeutral':
                new_node = IsNewNeutral()
            elif chosen == 'WillPlayerWinAfterN':
                new_node = WillPlayerWinAfterN()
            elif chosen == 'AreThereAvailableColumnsToPlay':
                new_node = AreThereAvailableColumnsToPlay()
            return new_node
        elif chosen_node == 'Times':
            acceptable_nodes = ['VarScalar', 'VarScalarFromArray', 'functions_scalars', 'Constant', 'Argmax', 'Not']
            chosen_left = random.choice(acceptable_nodes)
            chosen_right = random.choice(acceptable_nodes)
            # Left
            if chosen_left == 'VarScalar':
                chosen_node_left = self.finish_tree(node, 'VarScalar')
            elif chosen_left == 'VarScalarFromArray':
                chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
            elif chosen_left == 'Constant':
                chosen_node_left = self.finish_tree(node, 'Constant')
            elif chosen_left == 'Argmax':
                chosen_node_left = self.finish_tree(node, 'Argmax')
            elif chosen_left == 'Not':
                chosen_node_left = self.finish_tree(node, 'Not')
            else:
                chosen_node_left = self.finish_tree(node, 'functions_scalars')
            # Right
            if chosen_right == 'VarScalar':
                chosen_node_right = self.finish_tree(node, 'VarScalar')
            elif chosen_right == 'VarScalarFromArray':
                chosen_node_right = self.finish_tree(node, 'VarScalarFromArray')
            elif chosen_right == 'Constant':
                chosen_node_right = self.finish_tree(node, 'Constant')
            elif chosen_right == 'Argmax':
                chosen_node_right = self.finish_tree(node, 'Argmax')
            elif chosen_right == 'Not':
                chosen_node_right = self.finish_tree(node, 'Not')
            else:
                chosen_node_right = self.finish_tree(node, 'functions_scalars')
            new_node = Times(chosen_node_left, chosen_node_right)
            return new_node
        elif chosen_node == 'Plus':
            acceptable_nodes = ['VarScalar', 'VarScalarFromArray', 'functions_scalars',
                                'Times', 'Plus', 'Minus', 'Constant', 'Argmax', 'Not'
                            ]
            chosen_left = random.choice(acceptable_nodes)
            chosen_right = random.choice(acceptable_nodes)
            # Left
            if chosen_left == 'VarScalar':
                chosen_node_left = self.finish_tree(node, 'VarScalar')
            elif chosen_left == 'VarScalarFromArray':
                chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
            elif chosen_left == 'Constant':
                chosen_node_left = self.finish_tree(node, 'Constant')
            elif chosen_left == 'Argmax':
                chosen_node_left = self.finish_tree(node, 'Argmax')
            elif chosen_left == 'Not':
                chosen_node_left = self.finish_tree(node, 'Not')
            elif chosen_left == 'functions_scalars':
                chosen_node_left = self.finish_tree(node, 'functions_scalars')
            elif chosen_left == 'Times':
                chosen_node_left = self.finish_tree(node, 'Times')
            elif chosen_left == 'Plus':
                chosen_node_left = self.finish_tree(node, 'Plus')
            else:
                chosen_node_left = self.finish_tree(node, 'Minus')
            # Right
            if chosen_right == 'VarScalar':
                chosen_node_right = self.finish_tree(node, 'VarScalar')
            elif chosen_right == 'VarScalarFromArray':
                chosen_node_right = self.finish_tree(node, 'VarScalarFromArray')
            elif chosen_right == 'Constant':
                chosen_node_right = self.finish_tree(node, 'Constant')
            elif chosen_right == 'Argmax':
                chosen_node_right = self.finish_tree(node, 'Argmax')
            elif chosen_right == 'Not':
                chosen_node_right = self.finish_tree(node, 'Not')
            elif chosen_right == 'functions_scalars':
                chosen_node_right = self.finish_tree(node, 'functions_scalars')
            elif chosen_right == 'Times':
                chosen_node_right = self.finish_tree(node, 'Times')
            elif chosen_right == 'Plus':
                chosen_node_right = self.finish_tree(node, 'Plus')
            else:
                chosen_node_right = self.finish_tree(node, 'Minus')
            new_node = Plus(chosen_node_left, chosen_node_right)
            return new_node
        elif chosen_node == 'Minus':
            acceptable_nodes = ['VarScalar', 'VarScalarFromArray', 'functions_scalars',
                                'Times', 'Plus', 'Minus', 'Constant', 'Argmax', 'Not'
                            ]
            chosen_left = random.choice(acceptable_nodes)
            chosen_right = random.choice(acceptable_nodes)
            # Left
            if chosen_left == 'VarScalar':
                chosen_node_left = self.finish_tree(node, 'VarScalar')
            elif chosen_left == 'VarScalarFromArray':
                chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
            elif chosen_left == 'Constant':
                chosen_node_left = self.finish_tree(node, 'Constant')
            elif chosen_left == 'Argmax':
                chosen_node_left = self.finish_tree(node, 'Argmax')
            elif chosen_left == 'Not':
                chosen_node_left = self.finish_tree(node, 'Not')
            elif chosen_left == 'functions_scalars':
                chosen_node_left = self.finish_tree(node, 'functions_scalars')
            elif chosen_left == 'Times':
                chosen_node_left = self.finish_tree(node, 'Times')
            elif chosen_left == 'Plus':
                chosen_node_left = self.finish_tree(node, 'Plus')
            else:
                chosen_node_left = self.finish_tree(node, 'Minus')
            # Right
            if chosen_right == 'VarScalar':
                chosen_node_right = self.finish_tree(node, 'VarScalar')
            elif chosen_right == 'VarScalarFromArray':
                chosen_node_right = self.finish_tree(node, 'VarScalarFromArray')
            elif chosen_right == 'Constant':
                chosen_node_right = self.finish_tree(node, 'Constant')
            elif chosen_right == 'Argmax':
                chosen_node_right = self.finish_tree(node, 'Argmax')
            elif chosen_right == 'Not':
                chosen_node_right = self.finish_tree(node, 'Not')
            elif chosen_right == 'functions_scalars':
                chosen_node_right = self.finish_tree(node, 'functions_scalars')
            elif chosen_right == 'Times':
                chosen_node_right = self.finish_tree(node, 'Times')
            elif chosen_right == 'Plus':
                chosen_node_right = self.finish_tree(node, 'Plus')
            else:
                chosen_node_right = self.finish_tree(node, 'Minus')
            new_node = Minus(chosen_node_left, chosen_node_right)
            return new_node
        elif chosen_node == 'Sum':
            acceptable_nodes = ['VarList', 'Map']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'VarList':
                chosen_node = self.finish_tree(node, 'VarList')
            else:
                chosen_node = self.finish_tree(node, 'Map')
            new_node = Sum(chosen_node)
            return new_node
        elif chosen_node == 'Not':
            acceptable_nodes = ['boolean_function']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'boolean_function':
                chosen_node = self.finish_tree(node, 'boolean_function')
            new_node = Not(chosen_node)
            return new_node
        elif chosen_node == 'Map':
            # Function
            acceptable_nodes_1 = ['Function']
            # VarList
            acceptable_nodes_2 = ['VarList']
            chosen_left = random.choice(acceptable_nodes_1)
            chosen_right = random.choice(acceptable_nodes_2)
            if chosen_left == 'Function':
                chosen_node_left = self.finish_tree(node, 'Function')
            if chosen_right == 'VarList':
                chosen_node_right = self.finish_tree(node, 'VarList')
            new_node = Map(chosen_node_left, chosen_node_right)
            return new_node
        elif chosen_node == 'Function':
            acceptable_nodes = ['Times', 'Plus', 'Minus', 'Sum', 'Map', 'Function', 'Constant', 'Argmax']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'Times':
                chosen_node = self.finish_tree(node, 'Times')
            elif chosen == 'Plus':
                chosen_node = self.finish_tree(node, 'Plus')
            elif chosen == 'Minus':
                chosen_node = self.finish_tree(node, 'Minus')
            elif chosen == 'Sum':
                chosen_node = self.finish_tree(node, 'Sum')
            elif chosen == 'Map':
                chosen_node = self.finish_tree(node, 'Map')
            elif chosen == 'Constant':
                chosen_node = self.finish_tree(node, 'Constant')
            elif chosen == 'Argmax':
                chosen_node = self.finish_tree(node, 'Argmax')
            else:
                chosen_node = self.finish_tree(node, 'Function')
            new_node = Function(chosen_node)
            return new_node
        elif chosen_node == 'Argmax':
            acceptable_nodes = ['VarList', 'Map']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'VarList':
                chosen_node = self.finish_tree(node, 'VarList')
            elif chosen == 'Map':
                chosen_node = self.finish_tree(node, 'Map')
            new_node = Argmax(chosen_node)
            return new_node
        else:
            raise Exception('Unhandled DSL term at finish_tree. DSL term = ', chosen_node)

    def find_replacement(self, node):
        """
        Find a replacement for node according to the DSL. In this implementation,
        this node will also be deleted; therefore, it is needed to look at this
        node's parent and apply the appropriate changes. 
        """

        parent = node.parent

        # Check if HoleNode is the root
        if node.parent is None:
            acceptable_nodes = ['Argmax',
                                'Map',
                                'Sum', 
                                'Function',
                                'Plus',
                                'Minus',
                                'Times'
                            ]
            chosen_node = random.choice(acceptable_nodes)
            # Finish the tree with the chosen substitute
            new_node = self.finish_tree(node, chosen_node)
        # Check for Times
        elif parent.className() == 'Times':
            acceptable_nodes = ['VarScalar', 
                                'VarScalarFromArray',
                                'functions_scalars',
                                'Constant',
                                'Argmax',
                                'Not'
                            ]
            chosen_node = random.choice(acceptable_nodes)
            # Finish the tree with the chosen substitute
            new_node = self.finish_tree(node, chosen_node)
        elif parent.className() == 'Plus' or parent.className() == 'Minus':
            acceptable_nodes = ['VarScalar', 
                                'VarScalarFromArray', 
                                'functions_scalars',
                                'Constant',
                                'Argmax',
                                'Times',
                                'Plus',
                                'Minus',
                                'Not'
                                ]
            chosen_node = random.choice(acceptable_nodes)
            # Finish the tree with the chosen substitute
            new_node = self.finish_tree(node, chosen_node)
        elif parent.className() == 'Function':
            acceptable_nodes = ['Times', 
                                'Plus', 
                                'Minus', 
                                'Sum', 
                                'Map', 
                                'Function',
                                'Constant',
                                'Argmax'
                            ]
            chosen_node = random.choice(acceptable_nodes)
            # Finish the tree with the chosen substitute
            new_node = self.finish_tree(node, chosen_node)
        elif parent.className() == 'Argmax' or parent.className() == 'Sum':
            acceptable_nodes = ['VarList', 
                                'Map'
                            ]
            chosen_node = random.choice(acceptable_nodes)
            # Finish the tree with the chosen substitute
            new_node = self.finish_tree(node, chosen_node)
        elif parent.className() == 'Not':
            acceptable_nodes = ['boolean_function']
            chosen_node = random.choice(acceptable_nodes)
            # Finish the tree with the chosen substitute
            new_node = self.finish_tree(node, chosen_node)
        elif parent.className() == 'Map':
            # Map is a little different because it has two distinctive children,
            # so it must be done checked separately

            # Check the special case "HoleNode"
            if node.className() == 'HoleNode':
                chosen_node = random.choice(['Function', 'VarList'])
                if chosen_node == 'Function':
                    # Finish the tree with the chosen substitute
                    new_node = self.finish_tree(node, 'Function')
                else:
                    # It is a VarList
                    new_node = self.finish_tree(node, 'VarList')
            elif node.className() == 'Function':
                # Finish the tree with the chosen substitute
                new_node = self.finish_tree(node, 'Function')
            else:
                # It is a VarList
                new_node = self.finish_tree(node, 'VarList')
        else:
            raise Exception('Unhandled parent at find_replacement. parent = ', parent)

        # Add replacement to the original tree
        self.update_parent(new_node, None)
        return new_node


    def mutate(self, program):
        # Update the tree node's parents and children
        self.update_parent(program, None)
        self.curr_id = 0
        self.id_tree_nodes(program)
        max_tries = 1000
        while True:
            max_tries -= 1
            # Start from id 1 to not choose Argmax
            index_node = random.randint(0, self.curr_id-1)
            node_to_mutate = self.find_node(program, index_node)
            if isinstance(node_to_mutate, type(None)):
                print('Tree')
                program.print_tree()
                raise Exception('Node randomly selected does not exist.' + ' Index sampled = ' + str(index_node))
            if max_tries == 0:
                return program
            elif not node_to_mutate.can_mutate:
                continue
            else:
                break

        # Mutate it
        replacement = self.find_replacement(node_to_mutate)

        # Add replacement to the original tree
        if node_to_mutate.parent is not None:
            node_to_mutate.parent.children[:] = [replacement if child==node_to_mutate else child for child in node_to_mutate.parent.children]
        else:
            self.curr_id = 0
            self.id_tree_nodes(replacement)
            return replacement
        # Update the nodes ids. This is needed because when mutating a single 
        # node it is possible to create many other nodes (and not just one).
        # So a single id swap is not possible. This also prevents id "holes"
        # for the next mutation, this way every index sampled will be in the
        # tree.
        self.curr_id = 0
        self.id_tree_nodes(program)
        return program

    def finish_holes(self, initial_node):
        """ Finish HoleNodes randomly. """

        # Special case: root is a HoleNode
        if isinstance(initial_node, HoleNode):
            replacement = self.find_replacement(initial_node)
            replacement.parent = None
            return replacement
        else:
            list_of_nodes = self.get_traversal(initial_node)
            for node in list_of_nodes:
                if isinstance(node, HoleNode):
                    replacement = self.find_replacement(node)
                    # Add replacement to the original tree
                    node.parent.children[:] = [replacement if child==node else child for child in node.parent.children]
                else:
                    node.can_mutate = False
            return initial_node

    def get_random_program(self, state_1, state_2):
        self.update_parent(state_1, None)
        self.curr_id = 0
        self.id_tree_nodes(state_1)

        self.update_parent(state_2, None)
        self.curr_id = 0
        self.id_tree_nodes(state_2)
        state_1 = self.finish_holes(state_1)
        state_2 = self.finish_holes(state_2)

        # Update the children of the nodes in a top-down approach
        self.update_children(state_1)
        self.curr_id = 0
        self.id_tree_nodes(state_1)
        self.update_children(state_2)
        self.curr_id = 0
        self.id_tree_nodes(state_2)

        # Update the parent of the nodes
        self.update_parent(state_1, None)
        self.update_parent(state_2, None)

        return state_1, state_2

    def get_mutated_program(self, program_1, program_2):
        tree_to_mutate = random.randint(1, 2)

        if tree_to_mutate == 1: 
            # Make a copy of program_1
            mutated_program = pickle.loads(pickle.dumps(program_1, -1))
            # Mutate it
            mutated_program = self.mutate(mutated_program)
            # Update the children of the nodes in a top-down approach
            self.update_children(mutated_program)
            return mutated_program, program_2
        else:
            # Make a copy of program_2
            mutated_program = pickle.loads(pickle.dumps(program_2, -1))
            # Mutate it
            mutated_program = self.mutate(mutated_program)
            # Update the children of the nodes in a top-down approach
            self.update_children(mutated_program)
            return program_1, mutated_program

    def run(self, initial_state_1, initial_state_2, is_complete):
        """ 
        Main routine of Simulated Annealing. SA will start mutating from
        initial_state and the first player to be beaten is player_to_be_beaten.
        If this player is beaten, the player that beat it will be the one to be
        beaten in the next iteration. 
        """
        start_SA = time.time()
        start_state_1, start_state_2 = self.get_random_program(pickle.loads(pickle.dumps(initial_state_1, -1)), pickle.loads(pickle.dumps(initial_state_2, -1)))

        with open(self.log_file, 'a') as f:
            print('Initial SA program below', file=f)
            print(start_state_1.to_string(), file=f)
            print(start_state_2.to_string(), file=f)
            print('Initial time = ', self.initial_time, file=f)
        
        # Program to be beaten
        to_be_beaten = self.get_glenn_player()

        best_score = None
        best_program_1 = None
        best_program_2 = None

        time_limit_reached = False
        # Loop it until time limit is reached
        while True:
            with open(self.log_file, 'a') as f:
                print('Temperature is too low. Restarting it.\n', file=f)
            if time_limit_reached:
                break
            curr_temp = self.init_temp
            current_program_1 = pickle.loads(pickle.dumps(start_state_1, -1))
            current_program_2 = pickle.loads(pickle.dumps(start_state_2, -1))
            # Transform into a Player object
            current_player = self.get_object(current_program_1, current_program_2)

            # Evaluates the current_player against to_be_beaten
            try:
                v, l, d = self.triage_evaluation(current_player, to_be_beaten)
            except Exception as e:
                v = 0
                l = 0
                d = 0
            current_score = self.n_games - v

            if best_score is None or current_score < best_score:
                best_score = current_score
                best_program_1 = current_program_1
                best_program_2 = current_program_2

            iteration = 1
            # If temperature gets too low, reset it
            while curr_temp > 1:
                # Stop condition
                if time.time() - start_SA > self.max_time:
                    with open(self.log_file, 'a') as f:
                        print('Time limit reached', file=f)
                        print('Current time = ', time.time() - start_SA, file=f)
                        print('Max time = ', self.max_time, file=f)
                    time_limit_reached = True
                    break

                mutated_program_1, mutated_program_2 = self.get_mutated_program(current_program_1, current_program_2)
                mutated_player = self.get_object(mutated_program_1, mutated_program_2)

                # Evaluates the mutated_player against to_be_beaten
                try:
                    v, l, d = self.triage_evaluation(mutated_player, to_be_beaten)
                except Exception as e:
                    v = 0
                    l = 0
                    d = 0
                mutated_score = self.n_games - v

                if best_score is None or mutated_score < best_score:
                    best_score = mutated_score
                    best_program_1 = mutated_program_1
                    best_program_2 = mutated_program_2
                    elapsed_SA = time.time() - start_SA
                    self.time_elapsed.append(elapsed_SA + self.initial_time)
                    self.wins_vs_glenn_y.append(v)

                    with open(self.log_file, 'a') as f:
                        print('SA - Mutated player was better than best - V/L/D =', v, l, d, file=f)
                        print('Mutated program 1 = ', current_program_1.to_string(), file=f)
                        print('Mutated program 2 = ', current_program_2.to_string(), file=f)
                        print('Best program 1 = ', best_program_1.to_string(), file=f)
                        print('Best program 2 = ', best_program_2.to_string(), file=f)
                        print('Time elapsed so far = ', time.time() - start_SA + self.initial_time, file=f)
                        print(file=f)

                    # This is to avoid a "Permission Denied" error when trying to
                    # save an image too quickly between iteration, causing this error
                    if v != 0:
                        # Generath the graph against Glenn
                        self.generath_graph()

                    # Save graph data to file
                    with open(self.folder + 'graph_data', 'wb') as file:
                        pickle.dump([best_program_1, best_program_2, self.get_object(best_program_1, best_program_2), self.wins_vs_glenn_x, self.wins_vs_glenn_y, self.time_elapsed], file)
                    with open(self.folder + 'log_graph.txt', 'w') as f:
                        print('self.wins_vs_glenn_x = ', self.wins_vs_glenn_x, file=f)
                        print('self.wins_vs_glenn_y = ', self.wins_vs_glenn_y, file=f)
                        print('self.time_elapsed = ', self.time_elapsed, file=f)

                if self.acceptance_function(current_score, mutated_score, curr_temp):
                    current_score = mutated_score
                    current_program_1 = mutated_program_1
                    current_program_2 = mutated_program_2
                    

                curr_temp = self.temperature_schedule(iteration, curr_temp)
                iteration += 1

        # Add a final data manually for the graph to always end at self.max_time
        self.time_elapsed.append(self.max_time + self.initial_time)
        self.wins_vs_glenn_y.append(self.wins_vs_glenn_y[-1])

        # Save graph data to file
        with open(self.folder + 'graph_data', 'wb') as file:
            pickle.dump([best_program_1, best_program_2, self.get_object(best_program_1, best_program_2), self.wins_vs_glenn_x, self.wins_vs_glenn_y, self.time_elapsed], file)
        with open(self.folder + 'log_graph.txt', 'w') as f:
            print('self.wins_vs_glenn_x = ', self.wins_vs_glenn_x, file=f)
            print('self.wins_vs_glenn_y = ', self.wins_vs_glenn_y, file=f)
            print('self.time_elapsed = ', self.time_elapsed, file=f)

        with open(self.log_file, 'a') as f:
            print('SA ended', v, l, d, file=f)
            print('Best program 1 = ', best_program_1.to_string(), file=f)
            print('Best program 2 = ', best_program_2.to_string(), file=f)
            print('Time elapsed = ', time.time() - start_SA + self.initial_time, file=f)
            print(file=f)

        self.generath_graph()

        return best_program_1, best_program_2, self.get_object(best_program_1, best_program_2), self.wins_vs_glenn_x, self.wins_vs_glenn_y, self.time_elapsed


    def temperature_schedule(self, iteration, curr_temperature):
        """ Calculate the next temperature used for the score calculation. """

        return self.init_temp / (1 + 0.9*iteration)

    def acceptance_function(self, best_J, mutated_J, curr_temp):
        """ 
        Return a boolean representing whether the mutated program will be 
        accepted.
        """
        acceptance_term = np.exp((best_J - mutated_J) * (100 / curr_temp))
        chosen_value = min(1, acceptance_term)

        sampled_value = random.uniform(0, 1)
        if sampled_value <= chosen_value:
            return True
        else:
            return False

    def triage_evaluation(self, player_1, player_2):

        victories = 0
        losses = 0
        draws = 0

        # Play 5% of n_games
        first_step = math.floor(0.05 * self.n_games)
        v, l, d = self.evaluate(player_1, player_2, first_step)
        victories += v
        losses += l
        draws += d
        # If it does not win at least 30%, return
        if v < (30 * first_step) / 100:
            return victories, losses, draws

        # Play 35% of n_games
        second_step = math.floor(0.35 * self.n_games)
        v, l, d = self.evaluate(player_1, player_2, second_step)
        victories += v
        losses += l
        draws += d
        # If it does not win at least 40%, return
        if v < (40 * second_step) / 100:
            return victories, losses, draws

        # Play the remaining games
        third_step = math.floor(0.60 * self.n_games)
        v, l, d = self.evaluate(player_1, player_2, third_step)
        victories += v
        losses += l
        draws += d
        return victories, losses, draws


    def evaluate(self, player_1, player_2, n_games):
        """ 
        Play self.n_games Can't Stop games between player_1 and player_2. The
        players swap who is the first player per iteration because Can't Stop
        is biased towards the player who plays the first move.
        """

        victories = 0
        losses = 0
        draws = 0

        for i in range(n_games):
            game = Game(2, 4, 6, [2,12], 2, 2)
            if i%2 == 0:
                    who_won = simplified_play_single_game(
                                                        player_1, 
                                                        player_2, 
                                                        game, 
                                                        self.max_game_rounds
                                                    )
                    if who_won == 1:
                        victories += 1
                    elif who_won == 2:
                        losses += 1
                    else:
                        draws += 1
            else:
                who_won = simplified_play_single_game(
                                                    player_2, 
                                                    player_1, 
                                                    game, 
                                                    self.max_game_rounds
                                                )
                if who_won == 2:
                    victories += 1
                elif who_won == 1:
                    losses += 1
                else:
                    draws += 1

        return victories, losses, draws

    def generath_graph(self):

        plt.grid()
        X = [int(elem) for elem in self.time_elapsed]
        Y = self.wins_vs_glenn_y
        plt.xlabel('Time elapsed (s)')
        plt.ylabel('Victories')
        plt.ylim([0, self.n_games])
        plt.suptitle('SA - Games against Glenn')
        plt.plot(X,Y)
        plt.savefig(self.folder + 'vs_glenn' + '.jpg', dpi=1200)
        plt.close()

if __name__ == "__main__":

    incomplete = [
                    HoleNode(),
                    Argmax(HoleNode()),
                    Argmax(Map(HoleNode(), HoleNode())),
                    Argmax(Map(Function(HoleNode()), VarList('actions'))),
                    Argmax(Map(Function(Sum(HoleNode())), VarList('actions'))),
                    Argmax(Map(Function(Sum(Map(Function(HoleNode()), NoneNode()))), VarList('actions'))),
                    Argmax(Map(Function(Sum(Map(Function(Minus(Times(HoleNode(), HoleNode()), HoleNode())), NoneNode()))), VarList('actions'))),
                ]
    
    chosen = int(sys.argv[1])
    n_SA_iterations = 12
    max_game_rounds = 500
    # Stop condition for SA (in seconds)
    max_time = 300
    n_games = 1000
    init_temp = 2000
    d = 1
    initial_time = 0.0
    algo_name = 'SA_' + str(chosen)
    start_SA = time.time()
    SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name, initial_time, max_time)
    is_complete = False
    _, _, _, _, _, _ = SA.run(incomplete[chosen], incomplete[chosen], is_complete)
    end_SA = time.time() - start_SA
    print('Time elapsed = ', end_SA)
