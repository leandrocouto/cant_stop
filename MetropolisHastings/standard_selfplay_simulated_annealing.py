import math
import sys
import pickle
import time
import re
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,'..')
from play_game_template import simplified_play_single_game
from MetropolisHastings.parse_tree import ParseTree
from MetropolisHastings.DSL import DSL
from game import Game
from Script import Script
from players.rule_of_28_player import Rule_of_28_Player

class StandardSelfplaySimulatedAnnealing:
    """
    Simulated Annealing but instead of keeping a score on how many actions this
    algorithm got it correctly (when compared to an oracle), the score is now
    computed on how many victories the mutated get against the current program.
    The mutated program is accepted if it gets more victories than the current
    program playing against itself.
    """
    def __init__(self, n_iterations, tree_max_nodes, n_games, n_games_glenn, 
        max_game_rounds):
        """
        Metropolis Hastings with temperature schedule. This allows the 
        algorithm to explore more the space search.
        - n_games is the number of games played in selfplay.
        - n_games_glenn is the number of games played against Glenn's heuristic.
        - max_game_rounds is the number of rounds necessary in a game to
        consider it a draw. This is necessary because Can't Stop games can
        theoretically last forever.
        """

        self.n_iterations = n_iterations
        self.tree_max_nodes = tree_max_nodes
        self.n_games = n_games
        self.n_games_glenn = n_games_glenn
        self.max_game_rounds = max_game_rounds

        self.filename = 'standard_selfplay_' + str(self.n_iterations) + 'ite_' + \
        str(self.tree_max_nodes) + 'tree_' + str(self.n_games) + 'selfplay_' + \
        str(self.n_games_glenn) + 'glenn' 

        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        self.tree_string = ParseTree(DSL('S', True), self.tree_max_nodes)
        self.tree_column = ParseTree(DSL('S', False), self.tree_max_nodes)

        # For analysis
        self.victories = []
        self.losses = []
        self.draws = []


    def run(self):
        """ Main routine of the SA algorithm. """

        full_run = time.time()

        self.tree_string.build_tree(self.tree_string.root)
        self.tree_column.build_tree(self.tree_column.root)

        # Main loop
        for i in range(self.n_iterations):
            start = time.time()
            # Make a copy of the tree for future mutation
            new_tree_string = pickle.loads(pickle.dumps(self.tree_string, -1))
            new_tree_column = pickle.loads(pickle.dumps(self.tree_column, -1))

            new_tree_string.mutate_tree()
            new_tree_column.mutate_tree()

            current_program_string = self.tree_string.generate_program()
            current_program_column = self.tree_column.generate_program()
            
            mutated_program_string = new_tree_string.generate_program()
            mutated_program_column = new_tree_column.generate_program()

            script_best_player = self.generate_player(
                                                current_program_string, 
                                                current_program_column,
                                                i
                                                )
            script_mutated_player = self.generate_player(
                                                mutated_program_string,
                                                mutated_program_column,
                                                i
                                                )

            victories, losses, draws = self.selfplay(script_mutated_player, script_best_player)

            # If the new synthesized program is better
            if victories > losses:
                self.tree_string = new_tree_string
                self.tree_column = new_tree_column
                best_program_string = self.tree_string.generate_program()
                best_program_column = self.tree_column.generate_program()
                script_best_player = self.generate_player(
                                                        best_program_string,
                                                        best_program_column,
                                                        i
                                                        )
                vic, loss, draw = self.validate_against_glenn(script_best_player)
                self.victories.append(vic)
                self.losses.append(loss)
                self.draws.append(draw)
                with open(self.filename + '/' + 'standard_selfplay_log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, 'New program accepted - V/L/D against Glenn = ', vic, loss, draw, file=f)
                
            elapsed_time = time.time() - start
            with open(self.filename + '/' + 'standard_selfplay_log_' + self.filename + '.txt', 'a') as f:
                print('Iteration -', i, '- Elapsed time: ', elapsed_time, file=f)
        
        best_program_string = self.tree_string.generate_program()
        best_program_column = self.tree_column.generate_program()
        script_best_player = self.generate_player(
                                                best_program_string,
                                                best_program_column,
                                                i
                                                )

        full_run_elapsed_time = time.time() - full_run
        with open(self.filename + '/' + 'standard_selfplay_log_' + self.filename + '.txt', 'a') as f:
            print('Full program elapsed time = ', full_run_elapsed_time, file=f)

        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/' 

        script = Script(
                        best_program_string, 
                        best_program_column, 
                        self.n_iterations, 
                        self.tree_max_nodes
                    )
        
        self.generate_report(dir_path + self.filename)

        script.save_file_custom(dir_path, self.filename)

        return best_program_string, best_program_column, script_best_player, self.tree_string, self.tree_column

    def selfplay(self, mutated_player, current_player):

        victories = 0
        losses = 0
        draws = 0
        for i in range(self.n_games):
            game = game = Game(2, 4, 6, [2,12], 2, 2)
            if i%2 == 0:
                    who_won = simplified_play_single_game(
                                                        mutated_player, 
                                                        current_player, 
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
                                                    current_player, 
                                                    mutated_player, 
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

    def generate_player(self, program_string, program_column, iteration):
        """ Generate a Player object given the program string. """

        script = Script(
                        program_string, 
                        program_column, 
                        self.n_iterations, 
                        self.tree_max_nodes
                    )
        return self._string_to_object(script._generateTextScript(iteration))

    def _string_to_object(self, str_class, *args, **kwargs):
        """ Transform a program written inside str_class to an object. """

        exec(str_class)
        class_name = re.search("class (.*):", str_class).group(1).partition("(")[0]
        return locals()[class_name](*args, **kwargs)

    def validate_against_glenn(self, current_script):
        """ Validate current script against Glenn's heuristic player. """

        glenn = Rule_of_28_Player()

        victories = 0
        losses = 0
        draws = 0

        for i in range(self.n_games_glenn):
            game = game = Game(2, 4, 6, [2,12], 2, 2)
            if i%2 == 0:
                    who_won = simplified_play_single_game(
                                                        current_script, 
                                                        glenn, 
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
                                                    glenn, 
                                                    current_script, 
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


    def generate_report(self, filename):
        
        x = list(range(len(self.victories)))

        plt.plot(x, self.victories, color='green', label='Victory')
        plt.plot(x, self.losses, color='red', label='Loss')
        plt.plot(x, self.draws, color='gray', label='Draw')
        plt.legend(loc="best")
        plt.title("Standard selfplay generated script against Glenn's heuristic")
        plt.xlabel('Iterations')
        plt.ylabel('Number of games')
        plt.savefig(filename + '.png')

beta = 0.5
n_iterations = 10000
tree_max_nodes = 100
n_games = 100
n_games_glenn = 100
max_game_rounds = 500

SSSA = StandardSelfplaySimulatedAnnealing(
                                    n_iterations,
                                    tree_max_nodes,
                                    n_games,
                                    n_games_glenn,
                                    max_game_rounds
                                )
SSSA.run()