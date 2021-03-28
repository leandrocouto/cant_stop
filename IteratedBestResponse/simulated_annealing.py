import math
import sys
import pickle
import time
import os
import copy
from  random import choice
from DSL import *
from rule_of_28_sketch import Rule_of_28_Player_PS
sys.path.insert(0,'..')
from game import Game
from play_game_template import simplified_play_single_game

class SimulatedAnnealing:
    def __init__(self, n_SA_iterations, max_game_rounds, n_games, init_temp, d,
                to_log):
        self.n_SA_iterations = n_SA_iterations
        self.max_game_rounds = max_game_rounds
        self.n_games = n_games
        self.init_temp = init_temp
        self.d = d
        self.curr_id = 0
    
    def get_object(self, program):
        program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
        return Rule_of_28_Player_PS(program_yes_no, program)

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


    def update_parent_and_children(self, program, parent):
        program.add_parent_and_children(parent)
        for child in program.children:
            self.update_parent_and_children(child, program)

    def mutate(self, program):
        # Update the tree node's parents and children
        self.update_parent_and_children(program, None)
        self.curr_id = 0
        self.id_tree_nodes(program)
        print('mutated antes')
        program.print_tree(program, '  ')
        print()
        max_tries = 1000
        while True:
            max_tries -= 1
            # Start from id 1 to not choose Argmax
            index_node = random.randint(1, self.curr_id-1)
            print('index = ', index_node)
            node_to_mutate = self.find_node(program, index_node)
            print('no encontrado = ', node_to_mutate)
            if node_to_mutate == None:
                raise Exception('Node randomly selected does not exist.' + \
                    ' Index sampled = ' + str(index_node))
            elif max_tries == 0:
                return
            else:
                break

        print('node_to_mutate.parent =', node_to_mutate.parent)
        print('node_to_mutate.parent.children =', node_to_mutate.parent.children)
        exit()
        # delete its children and mutate it with new values
        node_to_mutate.children = []

        # Update the nodes ids. This is needed because when mutating a single 
        # node it is possible to create many other nodes (and not just one).
        # So a single id swap is not possible. This also prevents id "holes"
        # for the next mutation, this way every index sampled will be in the
        # tree.
        self.curr_id = 0
        self.id_tree_nodes(program)

        print('mutated dps')
        program.print_tree(program, '  ')
        print()
        self.finish_tree(node_to_mutate)
        print('mutated dps dps')
        program.print_tree(program, '  ')
        exit()
        # Updating again after (possibly) finishing expanding possibly not
        # expanded nodes.
        self.current_id = 0
        self._update_nodes_ids(self.root)

    def run(self, initial_state, player_to_be_beaten):
        """ 
        Main routine of Simulated Annealing. SA will start mutating from
        initial_state and the first player to be beaten is player_to_be_beaten.
        If this player is beaten, the player that beat it will be the one to be
        beaten in the next iteration. 
        """

        # Original program
        curr_state = pickle.loads(pickle.dumps(initial_state, -1))
        curr_player = self.get_object(initial_state)
        # Program to be beaten is given from selfplay
        best_tree = None
        best_player = pickle.loads(pickle.dumps(curr_player, -1))

        # Evaluate the starting program against the program from selfplay
        try:
            victories, _, _ = self.evaluate(curr_player, best_player)
        # If the program gives an error during evaluation, this program should
        # be 'discarded/bad' -> set victory to 0.
        except Exception as e:
            victories = 0
        
        print('victories = ', victories)
        best_score = victories
            
        # Number of "successful" iterations
        successful = 0 
        curr_temp = self.init_temp
        for i in range(2, self.n_SA_iterations + 2):
            start = time.time()
            #print('sa ite', i)
            # Make a copy of curr_state
            mutated_tree = pickle.loads(pickle.dumps(curr_state, -1))
            # Mutate it
            self.mutate(mutated_tree)
            # Get the object of the mutated program
            mutated_player = self.get_object(mutated_tree)
            # Evaluates the mutated program against best_program
            try:
                victories_mut, losses_mut, draws_mut = self.evaluate(mutated_player, curr_player)
            except Exception as e:
                victories_mut = 0
                losses_mut = 0
                draws_mut = 0

            print('v/l/d = ', victories_mut, losses_mut, draws_mut)
            exit()
            new_score = victories_mut

            # Update score given the temperature parameters
            updated_score_best, updated_score_mutated = self.update_score(
                                                                best_score, 
                                                                new_score, 
                                                                curr_temp
                                                            )
            if updated_score_mutated > updated_score_best:
                successful += 1

                elapsed = time.time() - start
                print('SA - Iteration = ',i, '- Mutated player was better - Victories =', victories_mut)
                best_score = new_score
                # Copy the trees
                best_tree = mutated_tree
                # Copy the script
                best_player = mutated_player
            # update temperature according to schedule
            else:
                elapsed = time.time() - start
                print('SA - Iteration = ',i, '- Mutated player was worse - Victories =', victories_mut)
            curr_temp = self.temperature_schedule(i)

        # It can happen that the mutated state is never better than 
        # player_to_be_beaten, therefore we set best_tree and best_player to the 
        # original initial_state.
        if best_tree is None:
            best_tree = curr_tree
            best_player = curr_player
        print('Successful iterations of this SA = ', successful, 'out of', self.n_SA_iterations, 'iterations.')
        return best_tree, best_player

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