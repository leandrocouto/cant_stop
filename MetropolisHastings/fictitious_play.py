import math
import sys
import pickle
import time
import os
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
from MetropolisHastings.parse_tree import ParseTree
from MetropolisHastings.DSL import DSL
from MetropolisHastings.two_weights_DSL import TwoWeightsDSL
from MetropolisHastings.shared_weights_DSL import SharedWeightsDSL
from game import Game
from sketch import Sketch
from algorithm import Algorithm
from play_game_template import simplified_play_single_game
from play_game_template import play_single_game
from play_game_template import play_solitaire_single_game

class FictitiousPlay(Algorithm):
    """
    Simulated Annealing but instead of keeping a score on how many actions this
    algorithm got it correctly (when compared to an oracle), the score is now
    computed on how many victories the mutated get against the current program.
    The mutated program is accepted if it gets more victories than the current
    program playing against itself.
    """
    def __init__(self, algo_id, n_iterations, n_SA_iterations, tree_max_nodes, 
        d, init_temp, n_games_evaluate, n_games_glenn, n_games_uct, 
        n_games_solitaire, uct_playouts, eval_step, max_game_rounds, 
        iteration_run, yes_no_dsl, column_dsl):
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

        self.algo_id = algo_id
        self.n_SA_iterations = n_SA_iterations
        self.d = d
        self.init_temp = init_temp
        self.n_games_evaluate = n_games_evaluate
        self.eval_step = eval_step
        self.iteration_run = iteration_run

        super().__init__(tree_max_nodes, n_iterations, n_games_glenn, 
                            n_games_uct, n_games_solitaire, uct_playouts,
                            max_game_rounds, yes_no_dsl, column_dsl
                        )

        self.filename = str(self.algo_id) + '_' + \
                        str(self.n_iterations) + 'ite_' + \
                        str(self.n_SA_iterations) + 'SAite_' + \
                        str(self.n_games_evaluate) + 'eval_' + \
                        str(self.n_games_glenn) + 'glenn_' + \
                        str(self.n_games_uct) + 'uct_' + \
                        str(self.n_games_solitaire) + 'solitaire_' + \
                        str(self.iteration_run) + 'run'


        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        # For analysis
        self.scores = []

    def run(self):

        full_run = time.time()
        p_tree_string = ParseTree(self.yes_no_dsl, self.tree_max_nodes)
        p_tree_column = ParseTree(self.column_dsl, self.tree_max_nodes)

        p_tree_string.build_tree(p_tree_string.root)
        p_tree_column.build_tree(p_tree_column.root)

        p_program_string = p_tree_string.generate_program()
        p_program_column = p_tree_column.generate_program()

        p = self.generate_player(p_program_string, p_program_column, 'p')

        br_set = [p]

        for i in range(self.n_iterations):
            start = time.time()
            br_tree_string, br_tree_column, br_p = self.simulated_annealing(p_tree_string, p_tree_column, br_set)

            elapsed_time = time.time() - start

            victories_p, losses_p, draws_p = self.evaluate(p, br_set)
            score_p = sum(victories_p) / len(victories_p)

            victories_br_p, losses_br_p, draws_br_p = self.evaluate(br_p, br_set)
            score_br_p = sum(victories_br_p) / len(victories_br_p)

            self.games_played += 2 * self.n_SA_iterations * self.n_games_evaluate * len(br_set)
            self.games_played_all.append(self.games_played)
            
            # if br_p is better, keep it
            if score_br_p > score_p:
                p_tree_string = br_tree_string
                p_tree_column = br_tree_column
                p = br_p
                br_set.append(br_p)
                self.victories.append(victories_br_p)
                self.losses.append(losses_br_p)
                self.draws.append(draws_br_p)
                self.scores.append(score_br_p)

                self.games_played_successful.append(self.games_played)

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
                    self.games_played_uct.append(self.games_played)
                    v_uct, l_uct, d_uct = self.validate_against_UCT(p)
                    self.victories_against_UCT.append(v_uct)
                    self.losses_against_UCT.append(l_uct)
                    self.draws_against_UCT.append(d_uct)
                elapsed_time_uct = time.time() - start_uct

                # Validate with Solitaire
                start_solitaire = time.time()
                avg_solitaire, std_solitaire = self.validate_solitaire(p)
                self.avg_rounds_solitaire.append(avg_solitaire)
                self.std_rounds_solitaire.append(std_solitaire)
                elapsed_time_solitaire = time.time() - start_solitaire

                # Save data file
                iteration_data = (
                                    self.victories,
                                    self.losses,
                                    self.draws,
                                    self.victories_against_glenn,
                                    self.losses_against_glenn,
                                    self.draws_against_glenn,
                                    self.victories_against_UCT,
                                    self.losses_against_UCT,
                                    self.draws_against_UCT,
                                    self.avg_rounds_solitaire,
                                    self.std_rounds_solitaire,
                                    len(br_set) - 1,
                                    score_br_p, score_p,
                                    self.games_played,
                                    self.games_played_successful,
                                    self.games_played_all,
                                    self.games_played_uct,
                                    p_tree_string, p_tree_column
                                )
                folder = self.filename + '/data/' 
                if not os.path.exists(folder):
                    os.makedirs(folder)
                with open(folder + 'datafile_iteration_' + str(i) , 'wb') as file:
                    pickle.dump(iteration_data, file)
                # Save current script
                dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/data/' 
                script = Sketch(
                                p_tree_string.generate_program(), 
                                p_tree_column.generate_program(), 
                                self.n_iterations, 
                                self.tree_max_nodes
                            )      
                script.save_file_custom(dir_path, self.filename + '_iteration_' + str(i))

                # Generate the graphs with current data
                self.generate_report()

                with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, 'New program accepted - V/L/D new script vs old = ',
                        victories_br_p, losses_br_p, draws_br_p, 
                        'V/L/D against Glenn = ', v_glenn, l_glenn, d_glenn, 
                        'V/L/D against UCT', self.uct_playouts, 'playouts = ', v_uct, l_uct, d_uct, 
                        'Avg and std in Solitaire = ', avg_solitaire, std_solitaire, 
                        'Games played = ', self.games_played,
                        file=f)
                    print('Iteration -', i, 'SA elapsed time = ', elapsed_time,
                        'Glenn elapsed time = ', elapsed_time_glenn, 
                        'UCT elapsed time = ', elapsed_time_uct, 
                        'Solitaire elapsed time = ', elapsed_time_solitaire,
                        'Total elapsed time = ', elapsed_time + elapsed_time_glenn + elapsed_time_uct + elapsed_time_solitaire, file=f)

            # The new script was not better, ignore this iteration
            else:
                with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, '- Elapsed time: ', elapsed_time, 'Games played = ', self.games_played, file=f)

        # Save the best script
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/'
        script = Sketch(
                        p_program_string, 
                        p_program_column, 
                        self.n_iterations, 
                        self.tree_max_nodes
                    )      
        script.save_file_custom(dir_path, self.filename + '_best_script')

        full_run_elapsed_time = time.time() - full_run
        with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
            print('Full program elapsed time = ', full_run_elapsed_time, file=f)

        return p_program_string, p_program_column, p, p_tree_string, p_tree_column

    def simulated_annealing(self, p_tree_string, p_tree_column, br_set):
        
        # Builds an initially random program
        best_solution_string_tree = ParseTree(self.yes_no_dsl, self.tree_max_nodes)
        best_solution_column_tree = ParseTree(self.column_dsl, self.tree_max_nodes)
        best_solution_string_tree.build_tree(best_solution_string_tree.root)
        best_solution_column_tree.build_tree(best_solution_column_tree.root)
        best_string = best_solution_string_tree.generate_program()
        best_column = best_solution_column_tree.generate_program()
        best_solution = self.generate_player(best_string, best_column, 'SA_best')

        # Evaluates this program against the set of scripts.
        victories, losses, draws = self.evaluate(best_solution, br_set)

        best_score = sum(victories) / len(victories)

        curr_temp = self.init_temp
        for i in range(2, self.n_SA_iterations + 2):
            start = time.time()
            # Make a copy of curr_p
            mutated_tree_string = pickle.loads(pickle.dumps(best_solution_string_tree, -1))
            mutated_tree_column = pickle.loads(pickle.dumps(best_solution_column_tree, -1))
            # Mutate it
            mutated_tree_string.mutate_tree()
            mutated_tree_column.mutate_tree()
            # Get the programs for each type of actions
            mutated_p_string = mutated_tree_string.generate_program()
            mutated_p_column = mutated_tree_column.generate_program()
            # Build the script
            mutated_p = self.generate_player(mutated_p_string, mutated_p_column, 'mutated_' + str(i))
            # Evaluates the mutated program against p
            victories_mut, _, _ = self.evaluate(mutated_p, br_set)

            new_score = sum(victories_mut) / len(victories_mut)

            # Update score given the temperature parameters
            updated_score_best, updated_score_mutated = self.update_score(best_score, new_score, curr_temp)
            
            if updated_score_mutated > updated_score_best:
                best_score = new_score
                # Copy the trees
                best_solution_string_tree = mutated_tree_string
                best_solution_column_tree = mutated_tree_column
                # Copy the script
                best_solution = mutated_p
            # update temperature according to schedule
            curr_temp = self.temperature_schedule(i)
            elapsed_time = time.time() - start
        return best_solution_string_tree, best_solution_column_tree, best_solution

    def update_score(self, score_best, score_mutated, curr_temp):
        """ 
        Update the score according to the current temperature. 
        """
        
        new_score_best = score_best**(1 / curr_temp)
        new_score_mutated = score_mutated**(1 / curr_temp)
        return new_score_best, new_score_mutated

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

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """

        return self.d/math.log(iteration)

    def generate_report(self):
        
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/' 
        filename = dir_path + self.filename

        vic = [sum(self.victories[i])/len(self.victories[i]) for i in range(len(self.victories))]
        loss = [sum(self.losses[i])/len(self.losses[i]) for i in range(len(self.losses))]
        plt.plot(self.games_played_successful, vic, color='green', label='Victory')
        plt.plot(self.games_played_successful, loss, color='red', label='Loss')
        plt.legend(loc="best")
        plt.title(str(self.algo_id) + " - Generated script against br_set (average values)")
        plt.xlabel('Games played')
        plt.ylabel('Rate')
        plt.savefig(filename + '_vs_br_set.png')
        plt.close()


        plt.plot(self.games_played_successful, self.victories_against_glenn, color='green')
        plt.title(str(self.algo_id) + " - Games against Glenn")
        plt.xlabel('Games played')
        plt.ylabel('Number of victories')
        plt.savefig(filename + '_vs_glenn.png')
        plt.close()

        for i in range(len(self.uct_playouts)):
            victories = [vic[i] for vic in self.victories_against_UCT]  
            plt.plot(self.games_played_uct, victories, label=str(self.uct_playouts[i]) + " playouts")
        plt.legend(loc="best")
        plt.title(str(self.algo_id) + " - Games against UCT")
        plt.xlabel('Games played')
        plt.ylabel('Number of victories')
        plt.savefig(filename + '_vs_UCT.png')
        plt.close()

        plt.errorbar(self.games_played_successful, self.avg_rounds_solitaire, yerr=self.std_rounds_solitaire, fmt='-')
        plt.title(str(self.algo_id) + " - Average rounds in Solitaire Can't Stop")
        plt.xlabel('Games played')
        plt.ylabel('Number of rounds')
        plt.savefig(filename + '_solitaire.png')
        plt.close()

if __name__ == "__main__":
    algo_id = 'SAFP'
    n_iterations = 20
    n_SA_iterations = 10
    tree_max_nodes = 100
    d = 1
    init_temp = 1
    n_games_evaluate = 100
    n_games_glenn = 1000
    n_games_uct = 3
    n_games_solitaire = 3
    uct_playouts = [2, 3, 4]
    eval_step = 1
    max_game_rounds = 500
    iteration_run = 0

    yes_no_dsl = DSL('S')
    yes_no_dsl.set_type_action(True)
    column_dsl = DSL('S')
    column_dsl.set_type_action(False)

    fictitious_play = FictitiousPlay(
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
                                        column_dsl
                                    )
    fictitious_play.run()

