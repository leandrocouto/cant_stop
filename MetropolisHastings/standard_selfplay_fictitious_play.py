import math
import sys
import pickle
import time
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
sys.path.insert(0,'..')
from MetropolisHastings.parse_tree import ParseTree
from MetropolisHastings.DSL import DSL
from game import Game
from Script import Script
from players.glenn_player import Glenn_Player
from players.vanilla_uct_player import Vanilla_UCT
from play_game_template import simplified_play_single_game
from play_game_template import play_single_game

from players.script_test import ScriptTest

class StandardSelfplayFictitiousPlay:
    """
    Simulated Annealing but instead of keeping a score on how many actions this
    algorithm got it correctly (when compared to an oracle), the score is now
    computed on how many victories the mutated get against the current program.
    The mutated program is accepted if it gets more victories than the current
    program playing against itself.
    """
    def __init__(self, n_iterations, tree_max_nodes, n_games_evaluate, 
        n_games_glenn, n_games_uct, uct_playouts, eval_step, max_game_rounds):
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
        self.n_games_evaluate = n_games_evaluate
        self.n_games_glenn = n_games_glenn
        self.n_games_uct = n_games_uct
        self.uct_playouts = uct_playouts
        self.eval_step = eval_step
        self.max_game_rounds = max_game_rounds

        self.filename = 'standard_selfplay_fp_' + str(self.n_iterations) + 'ite_' + \
        str(self.tree_max_nodes) + 'tree_' + str(self.n_games_evaluate) + 'eval_' + \
        str(self.n_games_glenn) + 'glenn' + str(self.n_games_uct) + \
        'uct'

        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        self.tree_string = ParseTree(DSL('S', True), self.tree_max_nodes)
        self.tree_column = ParseTree(DSL('S', False), self.tree_max_nodes)

        # For analysis
        self.victories = []
        self.losses = []
        self.draws = []

        # For analysis - Games against Glenn
        self.victories_against_glenn = []
        self.losses_against_glenn = []
        self.draws_against_glenn = []

        # For analysis - Games against UCT
        self.victories_against_UCT = []
        self.losses_against_UCT = []
        self.draws_against_UCT = []


    def run(self):
        """ Main routine of the SA algorithm. """

        full_run = time.time()

        self.tree_string.build_tree(self.tree_string.root)
        self.tree_column.build_tree(self.tree_column.root)
        current_program_string = self.tree_string.generate_program()
        current_program_column = self.tree_column.generate_program()
        script_best_player = self.generate_player(
                                                current_program_string, 
                                                current_program_column,
                                                0
                                                )

        br_set = [script_best_player]
        # Main loop
        for i in range(self.n_iterations):
            start = time.time()
            # Make a copy of the tree for future mutation
            new_tree_string = pickle.loads(pickle.dumps(self.tree_string, -1))
            new_tree_column = pickle.loads(pickle.dumps(self.tree_column, -1))
            new_tree_string.mutate_tree()
            new_tree_column.mutate_tree()    
            mutated_program_string = new_tree_string.generate_program()
            mutated_program_column = new_tree_column.generate_program()
            script_mutated_player = self.generate_player(
                                                mutated_program_string,
                                                mutated_program_column,
                                                i
                                                )

            victories_best, losses_best, draws_best = self.evaluate(script_best_player, br_set)
            victories_mut, losses_mut, draws_mut = self.evaluate(script_mutated_player, br_set)
            score_best = sum(victories_best) / len(victories_best)
            score_mutated = sum(victories_mut) / len(victories_mut)

            # If the new synthesized program is better against br_p
            if score_mutated > score_best:
                self.victories.append(victories_mut)
                self.losses.append(losses_mut)
                self.draws.append(draws_mut)

                # Copy mutated to best
                self.tree_string = pickle.loads(pickle.dumps(new_tree_string, -1))
                self.tree_column = pickle.loads(pickle.dumps(new_tree_column, -1))
                current_program_string = pickle.loads(pickle.dumps(mutated_program_string, -1))
                current_program_column = pickle.loads(pickle.dumps(mutated_program_column, -1))
                
                script_best_player = copy.deepcopy(script_mutated_player)
                
                # Add the new player to br_set
                br_set.append(script_mutated_player)

                start_glenn = time.time()
                v_glenn, l_glenn, d_glenn = self.validate_against_glenn(script_best_player)
                self.victories_against_glenn.append(v_glenn)
                self.losses_against_glenn.append(l_glenn)
                self.draws_against_glenn.append(d_glenn)
                elapsed_time_glenn = time.time() - start_glenn

                v_uct = None 
                l_uct = None 
                d_uct = None
                # Only play games against UCT every 10 successful iterations
                start_uct = time.time()
                
                if len(self.victories_against_glenn) % self.eval_step == 0:
                    v_uct, l_uct, d_uct = self.validate_against_UCT(script_best_player)
                    self.victories_against_UCT.append(v_uct)
                    self.losses_against_UCT.append(l_uct)
                    self.draws_against_UCT.append(d_uct)
                
                elapsed_time_uct = time.time() - start_uct

                elapsed_time = time.time() - start

                # Save data file
                iteration_data = (
                                    victories_mut, losses_mut, draws_mut,
                                    v_glenn, l_glenn, d_glenn,
                                    v_uct, l_uct, d_uct,
                                    self.tree_string, self.tree_column
                                )
                folder = self.filename + '/data/' 
                if not os.path.exists(folder):
                    os.makedirs(folder)
                with open(folder + 'datafile_iteration_' + str(i) , 'wb') as file:
                    pickle.dump(iteration_data, file)
                # Save current script
                dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/data/' 
                script = Script(
                                current_program_string, 
                                current_program_column,  
                                self.n_iterations, 
                                self.tree_max_nodes
                            )      
                script.save_file_custom(dir_path, self.filename + '_iteration_' + str(i))


                with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, 'New program accepted - ',
                        'V/L/D against br_set = ', victories_mut, losses_mut, draws_mut,
                        'V/L/D against Glenn = ', v_glenn, l_glenn, d_glenn, 
                        'V/L/D against UCT', self.uct_playouts, 'playouts = ', v_uct, l_uct, d_uct, 
                        file=f)
                    print('Iteration -', i, 'Glenn elapsed time = ', 
                        elapsed_time_glenn, 'UCT elapsed time = ', 
                        elapsed_time_uct, 'Total elapsed time = ', 
                        elapsed_time, file=f)
            else:
                elapsed_time = time.time() - start
                with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, '- Elapsed time: ', elapsed_time, file=f)
        
        best_program_string = self.tree_string.generate_program()
        best_program_column = self.tree_column.generate_program()
        script_best_player = self.generate_player(
                                                best_program_string,
                                                best_program_column,
                                                i
                                                )

        # Save the best script
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/'
        script = Script(
                        best_program_string, 
                        best_program_column, 
                        self.n_iterations, 
                        self.tree_max_nodes
                    )      
        script.save_file_custom(dir_path, self.filename + '_best_script')

        full_run_elapsed_time = time.time() - full_run
        with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
            print('Full program elapsed time = ', full_run_elapsed_time, file=f)

        self.generate_report()

        return best_program_string, best_program_column, script_best_player, self.tree_string, self.tree_column

    def evaluate(self, player, br_set):
        victories_rate = []
        losses_rate = []
        draws_rate = []
        for i in range(len(br_set)):
            v = 0
            l = 0
            d = 0
            for j in range(self.n_games_evaluate):
                game = Game(2, 4, 6, [2,12], 2, 2)
                if j%2 == 0:
                        who_won = simplified_play_single_game(
                                                            player, 
                                                            br_set[i], 
                                                            game, 
                                                            self.max_game_rounds
                                                        )
                        if who_won == 1:
                            v += 1
                        elif who_won == 2:
                            l += 1
                        else:
                            d += 1
                else:
                    who_won = simplified_play_single_game(
                                                        br_set[i], 
                                                        player, 
                                                        game, 
                                                        self.max_game_rounds
                                                    )
                    if who_won == 2:
                        v += 1
                    elif who_won == 1:
                        l += 1
                    else:
                        d += 1
            victories_rate.append(v / self.n_games_evaluate)
            losses_rate.append(l / self.n_games_evaluate)
            draws_rate.append(d / self.n_games_evaluate)

        return victories_rate, losses_rate, draws_rate

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

        glenn = Glenn_Player()

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

    def validate_against_UCT(self, current_script):
        """ Validate current script against UCT. """

        victories = []
        losses = []
        draws = []

        for i in range(len(self.uct_playouts)):
            v = 0
            l = 0
            d = 0
            for j in range(self.n_games_uct):
                game = game = Game(2, 4, 6, [2,12], 2, 2)
                uct = Vanilla_UCT(c = 1, n_simulations = self.uct_playouts[i])
                if j%2 == 0:
                        who_won = play_single_game(
                                                    current_script, 
                                                    uct, 
                                                    game, 
                                                    self.max_game_rounds
                                                    )
                        if who_won == 1:
                            v += 1
                        elif who_won == 2:
                            l += 1
                        else:
                            d += 1
                else:
                    who_won = play_single_game(
                                                uct, 
                                                current_script, 
                                                game, 
                                                self.max_game_rounds
                                                )
                    if who_won == 2:
                        v += 1
                    elif who_won == 1:
                        l += 1
                    else:
                        d += 1
            
            victories.append(v)
            losses.append(l)
            draws.append(d)

        return victories, losses, draws


    def generate_report(self):
        
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/' 
        filename = dir_path + self.filename

        x = list(range(len(self.victories)))

        vic = [sum(self.victories[i])/len(self.victories[i]) for i in range(len(self.victories))]
        loss = [sum(self.losses[i])/len(self.losses[i]) for i in range(len(self.losses))]
        draw = [sum(self.draws[i])/len(self.draws[i]) for i in range(len(self.draws))]

        plt.plot(x, vic, color='green', label='Victory')
        plt.plot(x, loss, color='red', label='Loss')
        plt.plot(x, draw, color='gray', label='Draw')
        plt.legend(loc="best")
        plt.title("Selfplay generated script against br_set (average values)")
        plt.xlabel('Iterations')
        plt.ylabel('Number of games')
        plt.savefig(filename + '_vs_br_set.png')

        plt.close()

        x = list(range(len(self.victories_against_glenn)))

        plt.plot(x, self.victories_against_glenn, color='green', label='Victory')
        plt.plot(x, self.losses_against_glenn, color='red', label='Loss')
        plt.plot(x, self.draws_against_glenn, color='gray', label='Draw')
        plt.legend(loc="best")
        plt.title("Fictitious Play - Games against Glenn")
        plt.xlabel('Iterations')
        plt.ylabel('Number of games')
        plt.savefig(filename + '_vs_glenn.png')

        plt.close()

        for i in range(len(self.uct_playouts)):
            victories = [vic[i] for vic in self.victories_against_UCT]  
            losses = [loss[i] for loss in self.losses_against_UCT]
            draws = [draw[i] for draw in self.draws_against_UCT]
            
            x = list(range(len(victories)))

            plt.plot(x, victories, color='green', label='Victory')
            plt.plot(x, losses, color='red', label='Loss')
            plt.plot(x, draws, color='gray', label='Draw')
            plt.legend(loc="best")
            plt.title("Standard Selfplay Fictitious Play - Games against UCT - " + str(self.uct_playouts[i]) + " playouts")
            plt.xlabel('Iterations')
            plt.ylabel('Number of games')
            plt.savefig(filename + '_vs_UCT_' + str(self.uct_playouts[i]) +'.png')

            plt.close()

if __name__ == "__main__":

    n_iterations = 10000
    tree_max_nodes = 100
    n_games_evaluate = 100
    n_games_glenn = 1000
    n_games_uct = 50
    max_game_rounds = 500
    uct_playouts = [50, 100, 200]
    eval_step = 10

    standard_selfplay_fp = StandardSelfplayFictitiousPlay(
                            n_iterations,
                            tree_max_nodes,
                            n_games_evaluate,
                            n_games_glenn,
                            n_games_uct,
                            uct_playouts,
                            eval_step,
                            max_game_rounds
                        )
    standard_selfplay_fp.run()