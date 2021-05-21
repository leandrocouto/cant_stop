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
    def __init__(self, n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name, initial_time):
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
        self.folder = algo_name + '_SA' + str(self.n_SA_iterations) + '/'
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

    def run(self, initial_state_1, initial_state_2, is_complete):
        """ 
        Main routine of Simulated Annealing. SA will start mutating from
        initial_state and the first player to be beaten is player_to_be_beaten.
        If this player is beaten, the player that beat it will be the one to be
        beaten in the next iteration. 
        """
        start_SA = time.time()

        self.update_parent(initial_state_1, None)
        self.curr_id = 0
        self.id_tree_nodes(initial_state_1)

        self.update_parent(initial_state_2, None)
        self.curr_id = 0
        self.id_tree_nodes(initial_state_2)

        # Original program
        curr_tree_1 = pickle.loads(pickle.dumps(initial_state_1, -1))
        curr_tree_2 = pickle.loads(pickle.dumps(initial_state_2, -1))
        if not is_complete:
            # Fill HoleNodes
            curr_tree_1 = self.finish_holes(curr_tree_1)
            curr_tree_2 = self.finish_holes(curr_tree_2)
        # Update the children of the nodes in a top-down approach
        self.update_children(curr_tree_1)
        self.curr_id = 0
        self.id_tree_nodes(curr_tree_1)

        self.update_children(curr_tree_2)
        self.curr_id = 0
        self.id_tree_nodes(curr_tree_2)

        with open(self.log_file, 'a') as f:
            print('Initial SA program below', file=f)
            print(curr_tree_1.to_string(), file=f)
            print(curr_tree_2.to_string(), file=f)
        # Update the parent of the nodes
        self.update_parent(curr_tree_1, None)
        self.update_parent(curr_tree_2, None)
        # Transform into a Player object
        curr_player = self.get_object(curr_tree_1, curr_tree_2)
        
        # Program to be beaten
        to_be_beaten = self.get_glenn_player()

        try:
            v, l, d = self.triage_evaluation(curr_player, to_be_beaten)
        except Exception as e:
            v = 0
            l = 0
            d = 0
        with open(self.log_file, 'a') as f:
            print('Initial script SA - V/L/D against Glenn = ', v, l, d, file=f)
            print(file=f)

        # Error of the best program
        best_J = self.n_games - v
        # Number of "successful" iterations
        successful = 0 
        curr_temp = self.init_temp
        for i in range(2, self.n_SA_iterations + 2):
            with open(self.log_file, 'a') as f:
                print('SA Iteration - ', i, file=f)
            start_ite = time.time()
            # Decides which tree to mutate
            tree_to_mutate = random.randint(1, 2)

            if tree_to_mutate == 1: 
                # Make a copy of curr_tree
                mutated_tree = pickle.loads(pickle.dumps(curr_tree_1, -1))
                # Mutate it
                mutated_tree = self.mutate(mutated_tree)
                # Update the children of the nodes in a top-down approach
                self.update_children(mutated_tree)
                # Get the object of the mutated program
                mutated_player = self.get_object(mutated_tree, curr_tree_2)
            else:
                # Make a copy of curr_tree
                mutated_tree = pickle.loads(pickle.dumps(curr_tree_2, -1))
                # Mutate it
                mutated_tree = self.mutate(mutated_tree)
                # Update the children of the nodes in a top-down approach
                self.update_children(mutated_tree)
                # Get the object of the mutated program
                mutated_player = self.get_object(curr_tree_1, mutated_tree)

            # Evaluates the mutated program against best_program
            try:
                victories_mut, losses_mut, draws_mut = self.triage_evaluation(mutated_player, to_be_beaten)
            except Exception as e:
                victories_mut = 0
                losses_mut = 0
                draws_mut = 0

            # Error of the best program
            mutated_J = self.n_games - victories_mut
            if victories_mut > 0:
                print('i = ', i, 'vic mut = ', victories_mut)
                print('mutated')
                if tree_to_mutate == 1:
                    print(mutated_tree.to_string())
                    print(curr_tree_2.to_string())
                else:
                    print(curr_tree_1.to_string())
                    print(mutated_tree.to_string())
                print('Time elapsed so far: ', time.time() - start_SA)
                print()
            if self.acceptance_function(best_J, mutated_J, curr_temp):
                successful += 1

                elapsed_ite = time.time() - start_ite
                elapsed_SA = time.time() - start_SA
                self.time_elapsed.append(self.initial_time + elapsed_SA)
                self.wins_vs_glenn_x.append(i)
                self.wins_vs_glenn_y.append(victories_mut)
                with open(self.log_file, 'a') as f:
                    print('SA - Iteration = ',i, '- Mutated player was better - V/L/D =', victories_mut, losses_mut, draws_mut, file=f)
                    print('Losses by best = ', best_J, file=f)
                    print('Losses by the mutated = ', mutated_J, file=f)
                    print('Current program 1 = ', curr_tree_1.to_string(), file=f)
                    print('Current program 2 = ', curr_tree_2.to_string(), file=f)
                    print('Mutated program (', tree_to_mutate, ') = ', mutated_tree.to_string(), file=f)
                    print('Time elapsed of this SA iteration = ', elapsed_ite, file=f)
                    print(file=f)

                # This is to avoid a "Permission Denied" error when trying to
                # save an image too quickly between iteration, causing this error
                if victories_mut != 0:
                    # Generath the graph against Glenn
                    self.generath_graph()
                # Copy the tree
                if tree_to_mutate == 1:
                    curr_tree_1 = mutated_tree
                else:
                    curr_tree_2 = mutated_tree
                # Copy the script
                curr_player = mutated_player
                # Copy the score
                best_J = mutated_J

                # Save graph data to file
                with open(self.folder + 'graph_data', 'wb') as file:
                    pickle.dump([curr_tree_1, curr_tree_2, curr_player, self.wins_vs_glenn_x, self.wins_vs_glenn_y, self.time_elapsed], file)
                with open(self.folder + 'log_graph.txt', 'w') as f:
                    print('self.wins_vs_glenn_x = ', self.wins_vs_glenn_x, file=f)
                    print('self.wins_vs_glenn_y = ', self.wins_vs_glenn_y, file=f)
                    print('self.time_elapsed = ', self.time_elapsed, file=f)
            # update temperature according to schedule
            else:
                elapsed_ite = time.time() - start_ite
                with open(self.log_file, 'a') as f:
                    print('SA - Iteration = ',i, '- Mutated player was worse - V/L/D =', victories_mut, losses_mut, draws_mut, file=f)
                    print('Current program 1 = ', curr_tree_1.to_string(), file=f)
                    print('Current program 2 = ', curr_tree_2.to_string(), file=f)
                    print('Mutated program (', tree_to_mutate, ') = ', mutated_tree.to_string(), file=f)
                    print('Time elapsed of this SA iteration = ', elapsed_ite, file=f)
                    print(file=f)
            curr_temp = self.temperature_schedule(i)
        with open(self.log_file, 'a') as f:
            print('self.wins_vs_glenn_x = ', self.wins_vs_glenn_x, file=f)
            print('self.wins_vs_glenn_y = ', self.wins_vs_glenn_y, file=f)
            print('self.time_elapsed = ', self.time_elapsed, file=f)
            print('Successful iterations of this SA = ', successful, 'out of', self.n_SA_iterations, 'iterations.', file=f)
            print('Total time elapsed of this SA = ', time.time() - start_SA, file=f)

        # Save graph data to file
        with open(self.folder + 'graph_data', 'wb') as file:
            pickle.dump([curr_tree_1, curr_tree_2, curr_player, self.wins_vs_glenn_x, self.wins_vs_glenn_y, self.time_elapsed], file)
        return curr_tree_1, curr_tree_2, curr_player, self.wins_vs_glenn_x, self.wins_vs_glenn_y, self.time_elapsed

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """

        return self.d/math.log(iteration)

    def acceptance_function(self, best_J, mutated_J, curr_temp):
        """ 
        Return a boolean representing whether the mutated program will be 
        accepted.
        """
        best_J = best_J * 100 / self.n_games
        mutated_J = mutated_J * 100 / self.n_games
        best_C = math.exp(-0.5 * best_J / curr_temp)
        mutated_C = math.exp(-0.5 * mutated_J / curr_temp)
        chosen_value = min(1, mutated_C / best_C)

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
        plt.suptitle('Games against Glenn (' + str(self.n_SA_iterations) + ' SA iterations)')
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
    n_SA_iterations = 200000
    max_game_rounds = 500
    n_games = 1000
    init_temp = 1
    d = 1
    initial_time = 0.0
    algo_name = 'SA_' + str(chosen)
    start_SA = time.time()
    SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name, initial_time)
    is_complete = False
    _, _, _, _, _, _ = SA.run(incomplete[chosen], incomplete[chosen], is_complete)
    end_SA = time.time() - start_SA
    print('Time elapsed = ', end_SA)
