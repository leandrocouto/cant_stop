import sys
sys.path.insert(0,'..')
import math
import pickle
import time
import os
import re
import matplotlib.pyplot as plt

from MetropolisHastings.DSL import DSL
from Script import Script
from game import Game
from MetropolisHastings.parse_tree import ParseTree
from players.rule_of_28_player import Rule_of_28_Player
from play_game_template import simplified_play_single_game

class GlennSimulatedAnnealing:
    def __init__(self, beta, n_iterations, tree_max_nodes, d, init_temp, 
        n_games, n_games_glenn, max_game_rounds):
        """
        Hybrid between selfplay and Simulated Annealing. Each procedure is 
        responsible for 50% of the score.
        """

        self.beta = beta
        self.n_iterations = n_iterations
        self.tree_max_nodes = tree_max_nodes
        self.d = d
        self.temperature = init_temp
        self.n_games = n_games
        self.n_games_glenn = n_games_glenn
        self.max_game_rounds = max_game_rounds

        self.filename = 'glenn_SA_' + str(self.n_iterations) + 'ite_' + \
        str(self.tree_max_nodes) + 'tree_' + str(self.n_games) + 'selfplay_' + \
        str(self.n_games_glenn) + 'glenn'

        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        self.tree_string = ParseTree(DSL('S', True), self.tree_max_nodes)
        self.tree_column = ParseTree(DSL('S', False), self.tree_max_nodes)

        # For analysis - Accuracy with oracle
        self.all_results = []
        self.passed_results = []
        # For analysis - Games against Glenn
        self.victories = []
        self.losses = []
        self.draws = []


    def run(self):
        """ Main routine of the SA algorithm. """

        full_run = time.time()

        self.tree_string.build_tree(self.tree_string.root)
        self.tree_column.build_tree(self.tree_column.root)

        # Main loop
        for i in range(2, self.n_iterations + 2):
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

            score_best, score_mutated  = self.calculate_score_function(
                                                        script_best_player, 
                                                        script_mutated_player
                                                        )
            # Instead of the classical SA that we divide the mutated score by
            # the current best program score (which we use the error rate), in
            # here we use best/mut because we are using the victory rate as
            # parameter for the score function.
            # Update score given the SA parameters
            score_best, score_mutated = self.update_score(score_best, score_mutated)

            # Accept program only if new score is higher.
            accept = min(1, score_best/score_mutated)
            
            # Adjust the temperature accordingly.
            self.temperature = self.temperature_schedule(i)

            # If the new synthesized program is better
            if accept == 1:
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
                with open(self.filename + '/' + 'glenn_SA_log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, 'New program accepted - V/L/D against Glenn = ', vic, loss, draw, file=f)
                
            elapsed_time = time.time() - start
            with open(self.filename + '/' + 'glenn_SA_log_' + self.filename + '.txt', 'a') as f:
                print('Iteration -', i, '- Elapsed time: ', elapsed_time, file=f)
        
        best_program_string = self.tree_string.generate_program()
        best_program_column = self.tree_column.generate_program()
        script_best_player = self.generate_player(
                                                best_program_string,
                                                best_program_column,
                                                i
                                                )

        full_run_elapsed_time = time.time() - full_run
        with open(self.filename + '/' + 'glenn_SA_log_' + self.filename + '.txt', 'a') as f:
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


    def update_score(self, score_best, score_mutated):
        """ 
        Update the score according to the current temperature. 
        """
        
        new_score_best = score_best**(1 / self.temperature)
        new_score_mutated = score_mutated**(1 / self.temperature)
        return new_score_best, new_score_mutated

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """
        return self.d/math.log(iteration)

    def calculate_score_function(self, script_best_player, script_mutated_player):

        vic_cur, _, _ = self.validate_against_glenn(script_best_player)
        vic_mut, _, _ = self.validate_against_glenn(script_mutated_player)
        victory_rate_glenn_cur = vic_cur / self.n_games_glenn
        victory_rate_glenn_mut = vic_mut / self.n_games_glenn

        score_cur = math.exp(-self.beta * victory_rate_glenn_cur)
        score_mut = math.exp(-self.beta * victory_rate_glenn_mut)

        return score_cur, score_mut

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
        plt.title("Glenn-SA generated script against Glenn's heuristic")
        plt.xlabel('Iteration')
        plt.ylabel('Number of games')
        plt.savefig(filename + '.png')


beta = 0.5
n_iterations = 10000
tree_max_nodes = 100
d = 1
init_temp = 1
n_games = 100
n_games_glenn = 100
max_game_rounds = 500

glenn_SA = GlennSimulatedAnnealing(
                                    beta,
                                    n_iterations,
                                    tree_max_nodes,
                                    d,
                                    init_temp,
                                    n_games,
                                    n_games_glenn,
                                    max_game_rounds
                                )

glenn_SA.run()