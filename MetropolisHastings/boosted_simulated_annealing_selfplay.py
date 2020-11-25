import math
import sys
import pickle
import time
import os
import matplotlib.pyplot as plt
import random
sys.path.insert(0,'..')
from MetropolisHastings.parse_tree import ParseTree
from game import Game
#from MetropolisHastings.DSL import DSL
#from sketch import Sketch
from MetropolisHastings.experimental_DSL import ExperimentalDSL
from experimental_sketch import Sketch
from simulated_annealing_selfplay import SimulatedAnnealingSelfplay
from play_game_template import simplified_play_single_game
from play_game_template import play_single_game

class BoostedSimulatedAnnealingSelfplay(SimulatedAnnealingSelfplay):
    """
    Simulated Annealing but instead of keeping a score on how many actions this
    algorithm got it correctly (when compared to an oracle), the score is now
    computed on how many victories the mutated get against the current program.
    The mutated program is accepted if it gets more victories than the current
    program playing against itself.
    """
    def __init__(self, algo_id, n_iterations, n_SA_iterations, 
        tree_max_nodes, d, init_temp, n_games_evaluate, n_games_glenn, 
        n_games_uct, n_games_solitaire, uct_playouts, eval_step, 
        max_game_rounds, iteration_run, yes_no_dsl, column_dsl, validate,
        scripts_to_collect):
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

        super().__init__(algo_id, n_iterations, n_SA_iterations, 
                        tree_max_nodes, d, init_temp, n_games_evaluate, 
                        n_games_glenn, n_games_uct, n_games_solitaire, 
                        uct_playouts, eval_step, max_game_rounds, 
                        iteration_run, yes_no_dsl, column_dsl, validate,
                        scripts_to_collect
                        )

        self.filename = str(self.algo_id) + '_' + \
                        str(self.n_iterations) + 'ite_' + \
                        str(self.n_SA_iterations) + 'SAite_' + \
                        str(self.n_games_evaluate) + 'eval_' + \
                        str(self.n_games_glenn) + 'glenn_' + \
                        str(self.n_games_uct) + 'uct_' + \
                        str(self.iteration_run) + 'run'

        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

    def simulated_annealing(self, p_tree_string, p_tree_column, p):
        
        curr_tree_string = pickle.loads(pickle.dumps(p_tree_string, -1))
        curr_tree_column = pickle.loads(pickle.dumps(p_tree_column, -1))
        curr_p_string = curr_tree_string.generate_program()
        curr_p_column = curr_tree_column.generate_program()
        curr_p = self.generate_player(curr_p_string, curr_p_column, 'SA_curr')

        # Evaluates this program against p.
        victories, losses, draws = self.evaluate(curr_p, p)
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
            new_score = victories_mut
            # if mutated_curr_p is better than p, then accept it
            if new_score > score:
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
        return best_solution_string_tree, best_solution_column_tree, best_solution

if __name__ == "__main__":
    algo_id = 'BSASP'
    n_iterations = 20
    n_SA_iterations = 10
    tree_max_nodes = 100
    d = 1
    init_temp = 1
    n_games_evaluate = 100
    n_games_glenn = 1000
    n_games_uct = 5
    n_games_solitaire = 3
    uct_playouts = [2, 3, 4]
    eval_step = 1
    max_game_rounds = 500
    iteration_run = 0
    validate = True
    scripts_to_collect = [100, 200, 500, 1000, 1500, 2000, 5000]

    '''
    yes_no_dsl = DSL('S')
    yes_no_dsl.set_type_action(True)
    column_dsl = DSL('S')
    column_dsl.set_type_action(False)
    '''
    yes_no_dsl = ExperimentalDSL('S')
    yes_no_dsl.set_type_action(True)
    column_dsl = ExperimentalDSL('S')
    column_dsl.set_type_action(False)

    boosted_selfplay_SA = BoostedSimulatedAnnealingSelfplay(
                                        algo_id,
                                        n_iterations,
                                        n_SA_iterations,
                                        tree_max_nodes,
                                        d,
                                        init_temp,
                                        n_games_evaluate,
                                        n_games_glenn,
                                        n_games_uct,
                                        n_games_solitaire,
                                        uct_playouts,
                                        eval_step,
                                        max_game_rounds,
                                        iteration_run,
                                        yes_no_dsl,
                                        column_dsl,
                                        validate,
                                        scripts_to_collect
                                    )
    boosted_selfplay_SA.run()

