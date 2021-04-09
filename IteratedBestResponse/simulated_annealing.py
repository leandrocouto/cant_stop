import math
import sys
import pickle
import time
import os
import copy
import random
import traceback
from DSL import *
from rule_of_28_sketch import Rule_of_28_Player_PS
sys.path.insert(0,'..')
from game import Game
from play_game_template import simplified_play_single_game

class SimulatedAnnealing:
    def __init__(self, n_SA_iterations, max_game_rounds, n_games, init_temp, d):
        self.n_SA_iterations = n_SA_iterations
        self.max_game_rounds = max_game_rounds
        self.n_games = n_games
        self.init_temp = init_temp
        self.d = d
        self.curr_id = 0
        self.wins_vs_glenn = []
    
    def get_object(self, program):
        program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
        return Rule_of_28_Player_PS(program_yes_no, program)

    def get_glenn(self):
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
        elif chosen_node == 'functions_scalars':
            acceptable_nodes = ['NumberAdvancedThisRound', 'NumberAdvancedByAction']#, 'IsNewNeutral']
            chosen = random.choice(acceptable_nodes)
            if chosen == 'NumberAdvancedThisRound':
                new_node = NumberAdvancedThisRound()
            elif chosen == 'NumberAdvancedByAction':
                new_node = NumberAdvancedByAction()
            else:
                new_node = IsNewNeutral()
            return new_node
        elif chosen_node == 'Times':
            acceptable_nodes = ['VarScalar', 'VarScalarFromArray', 'functions_scalars']
            chosen_left = random.choice(acceptable_nodes)
            chosen_right = random.choice(acceptable_nodes)
            # Left
            if chosen_left == 'VarScalar':
                chosen_node_left = self.finish_tree(node, 'VarScalar')
            elif chosen_left == 'VarScalarFromArray':
                chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
            else:
                chosen_node_left = self.finish_tree(node, 'functions_scalars')
            # Right
            if chosen_right == 'VarScalar':
                chosen_node_right = self.finish_tree(node, 'VarScalar')
            elif chosen_right == 'VarScalarFromArray':
                chosen_node_right = self.finish_tree(node, 'VarScalarFromArray')
            else:
                chosen_node_right = self.finish_tree(node, 'functions_scalars')
            new_node = Times(chosen_node_left, chosen_node_right)
            return new_node
        elif chosen_node == 'Plus':
            acceptable_nodes = ['VarScalar', 'VarScalarFromArray', 'functions_scalars',
                                'Times', 'Plus', 'Minus',
                            ]
            chosen_left = random.choice(acceptable_nodes)
            chosen_right = random.choice(acceptable_nodes)
            # Left
            if chosen_left == 'VarScalar':
                chosen_node_left = self.finish_tree(node, 'VarScalar')
            elif chosen_left == 'VarScalarFromArray':
                chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
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
                                'Times', 'Plus', 'Minus',
                            ]
            chosen_left = random.choice(acceptable_nodes)
            chosen_right = random.choice(acceptable_nodes)
            # Left
            if chosen_left == 'VarScalar':
                chosen_node_left = self.finish_tree(node, 'VarScalar')
            elif chosen_left == 'VarScalarFromArray':
                chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
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
            acceptable_nodes = ['Times', 'Plus', 'Minus', 'Sum', 'Map', 'Function']
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
            else:
                chosen_node = self.finish_tree(node, 'Function')
            new_node = Function(chosen_node)
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
        # Check for Times
        if parent.className() == 'Times':
            acceptable_nodes = ['VarScalar', 
                                'VarScalarFromArray',
                                'functions_scalars'
                            ]
            chosen_node = random.choice(acceptable_nodes)
            # Finish the tree with the chosen substitute
            new_node = self.finish_tree(node, chosen_node)
        elif parent.className() == 'Plus' or parent.className() == 'Minus':
            acceptable_nodes = ['VarScalar', 
                                'VarScalarFromArray', 
                                'functions_scalars',
                                #'Constant',
                                'Times',
                                'Plus',
                                'Minus'
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
                                'Function'
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
            index_node = random.randint(1, self.curr_id-1)
            node_to_mutate = self.find_node(program, index_node)
            if node_to_mutate == None:
                print('Tree')
                program.print_tree(program, '  ')
                raise Exception('Node randomly selected does not exist.' + ' Index sampled = ' + str(index_node))
            if not node_to_mutate.can_mutate:
                continue
            elif max_tries == 0:
                return
            else:
                break
        # Update the nodes ids. This is needed because when mutating a single 
        # node it is possible to create many other nodes (and not just one).
        # So a single id swap is not possible. This also prevents id "holes"
        # for the next mutation, this way every index sampled will be in the
        # tree.
        self.curr_id = 0
        self.id_tree_nodes(program)

        # Mutate it
        replacement = self.find_replacement(node_to_mutate)

        self.curr_id = 0
        self.id_tree_nodes(program)
        # Add replacement to the original tree
        node_to_mutate.parent.children[:] = [replacement if child==node_to_mutate else child for child in node_to_mutate.parent.children]
        # Update the node's ids
        self.curr_id = 0
        self.id_tree_nodes(program)
        return program

    def finish_holes(self, initial_node):
        """ Finish HoleNodes randomly. """

        list_of_nodes = self.get_traversal(initial_node)
        for node in list_of_nodes:
            if isinstance(node, HoleNode):
                replacement = self.find_replacement(node)
                # Add replacement to the original tree
                node.parent.children[:] = [replacement if child==node else child for child in node.parent.children]
            else:
                node.can_mutate = False

    def run(self, initial_state):
        """ 
        Main routine of Simulated Annealing. SA will start mutating from
        initial_state and the first player to be beaten is player_to_be_beaten.
        If this player is beaten, the player that beat it will be the one to be
        beaten in the next iteration. 
        """

        # In case the experiment is to fill HoleNodes, generate a script
        # that is compilable
        tries = 0
        while True:
            tries += 1
            # Original program
            curr_tree = pickle.loads(pickle.dumps(initial_state, -1))
            # Fill HoleNodes
            self.finish_holes(curr_tree)
            # Update the children of the nodes in a top-down approach
            self.update_children(curr_tree)
            self.curr_id = 0
            self.id_tree_nodes(curr_tree)
            print('programa inicial')
            print(curr_tree.toString())
            print('arvore inicial')
            curr_tree.print_tree()
            print()
            # Update the parent of the nodes
            self.update_parent(curr_tree, None)
            # Transform into a "playable" player
            curr_player = self.get_object(curr_tree)

            glenn = self.get_glenn()
            try:
                v, l, d = self.evaluate(curr_player, glenn)
                #The program is compilable, break the loop
                break
            except Exception as e:
                v = 0
                l = 0
                d = 0
        print('tries = ', tries)
        print('Initial script SA - V/L/D against Glenn = ', v, l, d)
        # Number of "successful" iterations
        successful = 0 
        curr_temp = self.init_temp
        for i in range(2, self.n_SA_iterations + 2):
            print('Iteration - ', i)
            start = time.time()
            # Make a copy of curr_tree
            mutated_tree = pickle.loads(pickle.dumps(curr_tree, -1))
            # Mutate it
            mutated_tree = self.mutate(mutated_tree)
            # Update the children of the nodes in a top-down approach
            self.update_children(mutated_tree)
            # Get the object of the mutated program
            mutated_player = self.get_object(mutated_tree)
            # Evaluates the mutated program against best_program
            try:
                victories_mut, losses_mut, draws_mut = self.evaluate(mutated_player, curr_player)
            except Exception as e:
                victories_mut = 0
                losses_mut = 0
                draws_mut = 0

            vic_needed = math.floor(0.55 * self.n_games)
            if victories_mut >= vic_needed:
                successful += 1

                elapsed = time.time() - start
                print('SA - Iteration = ',i, '- Mutated player was better - V/L/D =', victories_mut, losses_mut, draws_mut)
                print('Current program = ', curr_tree.toString())
                print('Mutated program = ', mutated_tree.toString())
                print('Mutated tree')
                mutated_tree.print_tree()
                glenn = self.get_glenn()
                v, l, d = self.evaluate(mutated_player, glenn)
                print('V/L/D against Glenn = ', v, l, d)
                self.wins_vs_glenn.append(v)
                print('self.wins_vs_glenn = ', self.wins_vs_glenn)
                # Copy the trees
                curr_tree = mutated_tree
                # Copy the script
                curr_player = mutated_player
                print()
            # update temperature according to schedule
            else:
                elapsed = time.time() - start
                print('SA - Iteration = ',i, '- Mutated player was worse - V/L/D =', victories_mut, losses_mut, draws_mut)
                print('Current program = ', curr_tree.toString())
                print('Mutated program = ', mutated_tree.toString())
                print()
            curr_temp = self.temperature_schedule(i)
        print('Successful iterations of this SA = ', successful, 'out of', self.n_SA_iterations, 'iterations.')
        return curr_tree, curr_player

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """

        return self.d/math.log(iteration)

    def update_score(self, score_best, score_mutated, curr_temp):
        """ Update the score according to the current temperature. """
        
        new_score_best = score_best**(1 / curr_temp)
        new_score_mutated = score_mutated**(1 / curr_temp)
        return new_score_best, new_score_mutated

    def evaluate(self, player_1, player_2):
        """ 
        Play self.n_games Can't Stop games between player_1 and player_2. The
        players swap who is the first player per iteration because Can't Stop
        is biased towards the player who plays the first move.
        """

        victories = 0
        losses = 0
        draws = 0

        for i in range(self.n_games):
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