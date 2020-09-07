import math
import sys
import pickle
import time
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import random
sys.path.insert(0,'..')
from play_game_template import simplified_play_single_game
from play_game_template import play_single_game
from MetropolisHastings.parse_tree import ParseTree
from MetropolisHastings.DSL import DSL
from game import Game
from Script import Script
from players.glenn_player import Glenn_Player
from players.vanilla_uct_player import Vanilla_UCT

class SelfplaySimulatedAnnealing:
    """
    Simulated Annealing but instead of keeping a score on how many actions this
    algorithm got it correctly (when compared to an oracle), the score is now
    computed on how many victories the mutated get against the current program.
    The mutated program is accepted if it gets more victories than the current
    program playing against itself.
    """
    def __init__(self, n_selfplay_iterations, n_SA_iterations, 
        tree_max_nodes, d, init_temp, n_games_evaluate, n_games_glenn, 
        n_games_uct, uct_playouts, eval_step, max_game_rounds):
        """
        Metropolis Hastings with temperature schedule. This allows the 
        algorithm to explore more the space search.
        - d is a constant for the temperature schedule.
        - init_temp is the temperature used for the first iteration. Following
          temperatures are calculated following self.temperature_schedule().
        - n_games is the number of games played in selfplay evaluation.
        - n_games_glenn is the number of games played against Glenn's heuristic.
        - max_game_rounds is the number of rounds necessary in a game to
        consider it a draw. This is necessary because Can't Stop games can
        theoretically last forever.
        """

        self.n_selfplay_iterations = n_selfplay_iterations
        self.n_SA_iterations = n_SA_iterations
        self.tree_max_nodes = tree_max_nodes
        self.d = d
        self.init_temp = init_temp
        self.n_games_evaluate = n_games_evaluate
        self.n_games_glenn = n_games_glenn
        self.n_games_uct = n_games_uct
        self.uct_playouts = uct_playouts
        self.eval_step = eval_step
        self.max_game_rounds = max_game_rounds

        self.filename = 'selfplay_SA_' + str(self.n_selfplay_iterations) + \
        'selfplay_ite_' + str(self.n_SA_iterations) + 'n_SA_ite_' + \
        str(self.tree_max_nodes) + 'tree_' + str(self.n_games_evaluate) + \
        'eval_' + str(self.n_games_glenn) + 'glenn_' + str(self.n_games_uct) + 'uct'

        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

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

    def selfplay(self):

        full_run = time.time()
        p_tree_string = ParseTree(DSL('S', True), self.tree_max_nodes)
        p_tree_column = ParseTree(DSL('S', False), self.tree_max_nodes)

        p_tree_string.build_tree(p_tree_string.root)
        p_tree_column.build_tree(p_tree_column.root)

        p_program_string = p_tree_string.generate_program()
        p_program_column = p_tree_column.generate_program()

        p = self.generate_player(p_program_string, p_program_column, 'p')

        for i in range(self.n_selfplay_iterations):
            start = time.time()
            br_tree_string, br_tree_column, br_p = self.simulated_annealing(
                                                    p_tree_string,
                                                    p_tree_column,
                                                    p)

            elapsed_time = time.time() - start
            
            victories, losses, draws = self.evaluate(br_p, p)
            # if br_p is better, keep it
            if victories > losses:
                p_tree_string = br_tree_string
                p_tree_column = br_tree_column
                p = br_p
                self.victories.append(victories)
                self.losses.append(losses)
                self.draws.append(draws)

                # Validade against Glenn's heuristic
                start_glenn = time.time()
                v_glenn, l_glenn, d_glenn = self.validate_against_glenn(p)
                self.victories_against_glenn.append(v_glenn)
                self.losses_against_glenn.append(l_glenn)
                self.draws_against_glenn.append(d_glenn)
                elapsed_time_glenn = time.time() - start_glenn
                # Validade against UCT
                start_uct = time.time()
                v_uct = None 
                l_uct = None 
                d_uct = None
                if len(self.victories_against_glenn) % self.eval_step == 0:
                    v_uct, l_uct, d_uct = self.validate_against_UCT(p)
                    self.victories_against_UCT.append(v_uct)
                    self.losses_against_UCT.append(l_uct)
                    self.draws_against_UCT.append(d_uct)
                elapsed_time_uct = time.time() - start_uct

                # Save data file
                iteration_data = (
                                    victories, losses, draws,
                                    v_glenn, l_glenn, d_glenn,
                                    v_uct, l_uct, d_uct,
                                    p_tree_string, p_tree_column
                                )
                folder = self.filename + '/data/' 
                if not os.path.exists(folder):
                    os.makedirs(folder)
                with open(folder + 'datafile_iteration_' + str(i) , 'wb') as file:
                    pickle.dump(iteration_data, file)
                # Save current script
                dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/data/' 
                script = Script(
                                p_tree_string.generate_program(), 
                                p_tree_column.generate_program(), 
                                self.n_selfplay_iterations, 
                                self.tree_max_nodes
                            )      
                script.save_file_custom(dir_path, self.filename + '_iteration_' + str(i))


                with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, 'New program accepted - ', 
                        'V/L/D new script vs old = ', victories, losses, draws, 
                        'V/L/D against Glenn = ', v_glenn, l_glenn, d_glenn, 
                        'V/L/D against UCT', self.uct_playouts, 'playouts = ', v_uct, l_uct, d_uct, 
                        file=f)
                    print('Iteration -', i, 'SA elapsed time = ', elapsed_time,
                        'Glenn elapsed time = ', elapsed_time_glenn, 
                        'UCT elapsed time = ', elapsed_time_uct, 
                        'Total elapsed time = ', elapsed_time + elapsed_time_glenn + elapsed_time_uct, file=f)

            # Otherwise stop the execution andr return the best player so far
            else:
                with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, '- Elapsed time: ', elapsed_time, file=f)

        # Save the best script
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/'
        script = Script(
                        p_program_string, 
                        p_program_column, 
                        self.n_selfplay_iterations, 
                        self.tree_max_nodes
                    )      
        script.save_file_custom(dir_path, self.filename + '_best_script')

        full_run_elapsed_time = time.time() - full_run
        with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
            print('Full program elapsed time = ', full_run_elapsed_time, file=f)

        self.generate_report()

        return p_program_string, p_program_column, p, p_tree_string, p_tree_column

    def simulated_annealing(self, p_tree_string, p_tree_column, p):
        
        # Builds an initially random program (curr)
        curr_tree_string = ParseTree(DSL('S', True), self.tree_max_nodes)
        curr_tree_column = ParseTree(DSL('S', False), self.tree_max_nodes)
        curr_tree_string.build_tree(curr_tree_string.root)
        curr_tree_column.build_tree(curr_tree_column.root)
        curr_p_string = curr_tree_string.generate_program()
        curr_p_column = curr_tree_column.generate_program()
        curr_p = self.generate_player(curr_p_string, curr_p_column, 'SA_curr')

        # Evaluates this program against p.
        victories, losses, draws = self.evaluate(curr_p, p)
        #print('initial eval = ', victories, losses, draws)
        score = victories
        best_score = score
        # Initially assumes that p is the best script of all
        best_solution_string_tree = p_tree_string
        best_solution_column_tree = p_tree_column
        best_solution = p

        curr_temp = self.init_temp

        for i in range(2, self.n_SA_iterations + 2):
            start = time.time()
            # Make a copy of curr_p
            mutated_tree_string = pickle.loads(pickle.dumps(curr_tree_string, -1))
            mutated_tree_column = pickle.loads(pickle.dumps(curr_tree_column, -1))
            # Mutate it
            mutated_tree_string.mutate_tree()
            mutated_tree_column.mutate_tree()
            # Get the programs for each type of actions
            mutated_curr_p_string = mutated_tree_string.generate_program()
            mutated_curr_p_column = mutated_tree_column.generate_program()
            # Build the script
            mutated_curr_p = self.generate_player(mutated_curr_p_string, mutated_curr_p_column, 'mutated_curr_' + str(i))
            # Evaluates the mutated program against p
            victories_mut, losses_mut, draws_mut = self.evaluate(mutated_curr_p, p)
            #print('iteration', i, ' eval = ', victories_mut, losses_mut, draws_mut)
            #print('Iteration', i, 'of SA - V/L/D = ', victories_mut, losses_mut, draws_mut)
            new_score = victories_mut
            #print('best score = ', best_score, ', new_score = ', new_score, ', score = ', score)
            # if mutated_curr_p is better than p, then accept it
            if new_score > score:
                #print('new score bateu score')
                score = new_score
                # Copy the trees
                curr_tree_string = mutated_tree_string 
                curr_tree_column = mutated_tree_column
                # Copy the programs
                curr_p_string = mutated_curr_p_string
                curr_p_column = mutated_curr_p_column
                # Copy the script
                curr_p = mutated_curr_p
                # Keep track of the best solution
                if new_score > best_score:
                    #print('new score bateu best score')
                    best_score = new_score
                    # Copy the trees
                    best_solution_string_tree = mutated_tree_string
                    best_solution_column_tree = mutated_tree_column
                    # Copy the script
                    best_solution = mutated_curr_p
            # even if not better, there is a chance to accept it
            else:
                delta = math.exp(-(score-new_score)/curr_temp)
                if(random.random() < delta):
                    score = new_score
                    # Copy the trees
                    curr_tree_string = mutated_tree_string 
                    curr_tree_column = mutated_tree_column
                    # Copy the programs
                    curr_p_string = mutated_curr_p_string
                    curr_p_column = mutated_curr_p_column
                    # Copy the script
                    curr_p = mutated_curr_p
            # update temperature according to schedule
            curr_temp = self.temperature_schedule(i)
            elapsed_time = time.time() - start
            #print('tempo da iteracao = ', elapsed_time)
        return best_solution_string_tree, best_solution_column_tree, best_solution

    def evaluate(self, first_player, second_player):
        victories = 0
        losses = 0
        draws = 0
        for i in range(self.n_games_evaluate):
            game = Game(2, 4, 6, [2,12], 2, 2)
            if i%2 == 0:
                    who_won = simplified_play_single_game(
                                                        first_player, 
                                                        second_player, 
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
                                                    second_player, 
                                                    first_player, 
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

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """

        return self.d/math.log(iteration)

    def generate_player(self, program_string, program_column, iteration):
        """ Generate a Player object given the program string. """

        script = Script(
                        program_string, 
                        program_column, 
                        self.n_selfplay_iterations, 
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

        plt.plot(x, self.victories, color='green', label='Victory')
        plt.plot(x, self.losses, color='red', label='Loss')
        plt.plot(x, self.draws, color='gray', label='Draw')
        plt.legend(loc="best")
        plt.title("Selfplay generated script against previous script")
        plt.xlabel('Iterations')
        plt.ylabel('Number of games')
        plt.savefig(filename + '_vs_previous_script.png')

        plt.close()

        x = list(range(len(self.victories_against_glenn)))

        plt.plot(x, self.victories_against_glenn, color='green', label='Victory')
        plt.plot(x, self.losses_against_glenn, color='red', label='Loss')
        plt.plot(x, self.draws_against_glenn, color='gray', label='Draw')
        plt.legend(loc="best")
        plt.title("Levi Selfplay SA - Games against Glenn")
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
            plt.title("Levi Selfplay SA - Games against UCT - " + str(self.uct_playouts[i]) + " playouts")
            plt.xlabel('Iterations')
            plt.ylabel('Number of games')
            plt.savefig(filename + '_vs_UCT_' + str(self.uct_playouts[i]) +'.png')

            plt.close()

if __name__ == "__main__":
    n_selfplay_iterations = 10
    n_SA_iterations = 5
    tree_max_nodes = 100
    d = 1
    init_temp = 1
    n_games_evaluate = 100
    n_games_glenn = 1000
    n_games_uct = 5
    uct_playouts = [2, 3, 4]
    eval_step = 1
    max_game_rounds = 500

    selfplay_SA = SelfplaySimulatedAnnealing(
                                        n_selfplay_iterations,
                                        n_SA_iterations,
                                        tree_max_nodes,
                                        d,
                                        init_temp,
                                        n_games_evaluate,
                                        n_games_glenn,
                                        n_games_uct,
                                        uct_playouts,
                                        eval_step,
                                        max_game_rounds
                                    )
    selfplay_SA.selfplay()

