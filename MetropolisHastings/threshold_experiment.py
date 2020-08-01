import sys
sys.path.insert(0,'..')
import math
import copy
from game import Game
from play_game_template import play_single_game
from players.vanilla_uct_player import Vanilla_UCT
from players.uct_player import UCTPlayer
from players.random_player import RandomPlayer
from players.rule_of_28_player import Rule_of_28_Player
from MetropolisHastings.parse_tree import ParseTree, Node
from MetropolisHastings.DSL import DSL
from MetropolisHastings.metropolis_hastings import MetropolisHastings
from MetropolisHastings.simulated_annealing import SimulatedAnnealing
from play_game_template import simplified_play_single_game, play_single_game
from Script import Script
import time
import pickle
import os.path
from random import sample
import numpy as np
from itertools import zip_longest
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from pylatex import Document, Section, Figure, NoEscape
from pylatex.utils import bold
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class ThresholdExperiment:
    def __init__(self, beta, n_games, iterations, batch_iterations, k, 
        thresholds, tree_max_nodes, n_cores, use_SA, d, init_temp, 
        n_games_glenn, dataset_name):

        self.beta = beta
        self.n_games = n_games
        self.iterations = iterations
        self.batch_iterations = batch_iterations
        self.k = k
        self.thresholds = thresholds
        self.tree_max_nodes = tree_max_nodes
        self.n_cores = n_cores
        self.use_SA = use_SA
        self.d = d
        self.init_temp = init_temp
        self.n_games_glenn = n_games_glenn
        self.dataset_name = dataset_name

        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
        
        if self.use_SA:    
            self.folder = dir_path + 'result_' + str(self.iterations) + \
                        'i_' +str(self.batch_iterations) +'b_' + \
                        str(self.tree_max_nodes) + 'n_SA/'
        else:
            self.folder = dir_path + 'result_' + str(self.iterations) + \
                        'i_' +str(self.batch_iterations) +'b_' + \
                        str(self.tree_max_nodes) + 'n_MH/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Single experiment
        self.total_errors_passed_results = []
        self.yes_errors_passed_results = []
        self.no_errors_passed_results = []
        self.numeric_errors_passed_results = []
        self.total_errors_all_results = []
        self.yes_errors_all_results = []
        self.no_errors_all_results = []
        self.numeric_errors_all_results = []

        # Batch experiment
        self.mean_total_errors_passed_results = {}
        self.mean_yes_errors_passed_results = {}
        self.mean_no_errors_passed_results = {}
        self.mean_numeric_errors_passed_results = {}
        self.mean_total_errors_all_results = {}
        self.mean_yes_errors_all_results = {}
        self.mean_no_errors_all_results = {}
        self.mean_numeric_errors_all_results = {}
        self.std_total_errors_passed_results = {}
        self.std_yes_errors_passed_results = {}
        self.std_no_errors_passed_results = {}
        self.std_numeric_errors_passed_results = {}
        self.std_total_errors_all_results = {}
        self.std_yes_errors_all_results = {}
        self.std_no_errors_all_results = {}
        self.std_numeric_errors_all_results = {}
        # Games against Glenn's heuristic
        self.victories = {}
        self.draws = {}
        self.losses = {}
        self.std_victories = {}
        self.std_draws = {}
        self.std_losses = {}

        # Add threshold keys for the dicts
        for threshold in self.thresholds:
            self.mean_total_errors_passed_results[threshold] = []
            self.mean_yes_errors_passed_results[threshold] = []
            self.mean_no_errors_passed_results[threshold] = []
            self.mean_numeric_errors_passed_results[threshold] = []
            self.mean_total_errors_all_results[threshold] = []
            self.mean_yes_errors_all_results[threshold] = []
            self.mean_no_errors_all_results[threshold] = []
            self.mean_numeric_errors_all_results[threshold] = []
            self.std_total_errors_passed_results[threshold] = []
            self.std_yes_errors_passed_results[threshold] = []
            self.std_no_errors_passed_results[threshold] = []
            self.std_numeric_errors_passed_results[threshold] = []
            self.std_total_errors_all_results[threshold] = []
            self.std_yes_errors_all_results[threshold] = []
            self.std_no_errors_all_results[threshold] = []
            self.std_numeric_errors_all_results[threshold] = []
            self.victories[threshold] = 0
            self.draws[threshold] = 0
            self.losses[threshold] = 0
            self.std_victories[threshold] = 0
            self.std_draws[threshold] = 0
            self.std_losses[threshold] = 0
  
        self.data_distribution = []

    def info_header_latex(self, doc):
        """Create a header used before each graph in the report."""

        doc.append(bold('Action distribution for different thresholds: '))
        for i in range(len(self.thresholds)):
            n_yes_actions = self.data_distribution[i][0]
            n_no_actions = self.data_distribution[i][1]
            n_num_actions = self.data_distribution[i][2]
            n_actions = n_yes_actions + n_no_actions + n_num_actions
            yes_percentage = round((n_yes_actions / n_actions) * 100, 2)
            no_percentage = round((n_no_actions / n_actions) * 100, 2)
            num_percentage = round((n_num_actions / n_actions) * 100, 2)
            doc.append(bold('\nThreshold = '))
            doc.append(str(self.thresholds[i]))
            doc.append(bold(' Total actions = '))
            doc.append(str(n_actions))
            doc.append(bold("  'Yes' = "))
            doc.append(str(n_yes_actions))
            doc.append("  (")
            doc.append(str(yes_percentage))
            doc.append("%)")
            doc.append(bold("  'No' = "))
            doc.append(str(n_no_actions))
            doc.append("  (")
            doc.append(str(no_percentage))
            doc.append("%)")
            doc.append(bold("  Num = "))
            doc.append(str(n_num_actions))
            doc.append("  (")
            doc.append(str(num_percentage))
            doc.append("%)")


    def generate_batch_report(self):

        matplotlib.use('Agg') 

        geometry_options = {"right": "2cm", "left": "2cm"}
        
        if self.use_SA:
            pdf_name = self.folder + 'result_' + str(self.iterations) + \
                        'i_' +str(self.batch_iterations) +'b_' + \
                        str(self.tree_max_nodes) + 'n_SA'
        else:
            pdf_name = self.folder + 'result_' + str(self.iterations) + \
                        'i_' +str(self.batch_iterations) +'b_' + \
                        str(self.tree_max_nodes) + 'n_MH'

        doc = Document(pdf_name, geometry_options=geometry_options)

        if self.use_SA:
            xlabel = 'SA Iterations'
        else:
            xlabel = 'MH Iterations'

        # Total error passed results
        with doc.create(Section('Total error rate - Passed results')):
            self.info_header_latex(doc)
            x = []
            y = []
            for threshold in self.thresholds:
                x.append(np.array(range(1, len(self.mean_total_errors_passed_results[threshold]) + 1)))
                y.append(np.array(self.mean_total_errors_passed_results[threshold]))
            _, ax = plt.subplots()
            for i in range(len(self.thresholds)):
                ax.errorbar(
                    x[i], 
                    y[i], 
                    yerr=self.std_total_errors_passed_results[self.thresholds[i]], 
                    label=self.thresholds[i], 
                    errorevery=5
                    )
            ax.legend(loc="best")
            ax.set(xlabel=xlabel, ylabel='Error rate', 
            title='Total error rate - Passed results')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Yes errors passed results
        with doc.create(Section(" 'Yes' action error rate - Passed results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for threshold in self.thresholds:
                x.append(np.array(range(1, len(self.mean_yes_errors_passed_results[threshold]) + 1)))
                y.append(np.array(self.mean_yes_errors_passed_results[threshold]))
            _, ax = plt.subplots()
            for i in range(len(self.thresholds)):
                ax.errorbar(
                    x[i], 
                    y[i], 
                    yerr=self.std_yes_errors_passed_results[self.thresholds[i]], 
                    label=self.thresholds[i], 
                    errorevery=5
                    )
            ax.legend(loc="best")
            ax.set(xlabel=xlabel, ylabel='Error rate', 
            title=" 'Yes' action error rate - Passed results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # No errors passed results
        with doc.create(Section(" 'No' action error rate - Passed results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for threshold in self.thresholds:
                x.append(np.array(range(1, len(self.mean_no_errors_passed_results[threshold]) + 1)))
                y.append(np.array(self.mean_no_errors_passed_results[threshold]))
            _, ax = plt.subplots()
            for i in range(len(self.thresholds)):
                ax.errorbar(
                    x[i], 
                    y[i], 
                    yerr=self.std_no_errors_passed_results[self.thresholds[i]], 
                    label=self.thresholds[i], 
                    errorevery=5
                    )
            ax.legend(loc="best")
            ax.set(xlabel=xlabel, ylabel='Error rate', 
            title=" 'No' action error rate - Passed results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Numeric errors passed results
        with doc.create(Section("Numeric action error rate - Passed results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for threshold in self.thresholds:
                x.append(np.array(range(1, len(self.mean_numeric_errors_passed_results[threshold]) + 1)))
                y.append(np.array(self.mean_numeric_errors_passed_results[threshold]))
            _, ax = plt.subplots()
            for i in range(len(self.thresholds)):
                ax.errorbar(
                    x[i], 
                    y[i], 
                    yerr=self.std_numeric_errors_passed_results[self.thresholds[i]], 
                    label=self.thresholds[i], 
                    errorevery=5
                    )
            ax.legend(loc="best")
            ax.set(xlabel=xlabel, ylabel='Error rate', 
            title="Numeric action error rate - Passed results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Total error all results
        with doc.create(Section('Total error rate - All results')):
            self.info_header_latex(doc)
            x = []
            y = []
            for threshold in self.thresholds:
                x.append(np.array(range(1, len(self.mean_total_errors_all_results[threshold]) + 1)))
                y.append(np.array(self.mean_total_errors_all_results[threshold]))
            _, ax = plt.subplots()
            for i in range(len(self.thresholds)):
                ax.errorbar(
                    x[i], 
                    y[i], 
                    yerr=self.std_total_errors_all_results[self.thresholds[i]], 
                    label=self.thresholds[i], 
                    errorevery=5
                    )
            ax.legend(loc="best")
            ax.set(xlabel=xlabel, ylabel='Error rate', 
            title='Total error rate - All results')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Yes errors all results
        with doc.create(Section(" 'Yes' action error rate - All results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for threshold in self.thresholds:
                x.append(np.array(range(1, len(self.mean_yes_errors_all_results[threshold]) + 1)))
                y.append(np.array(self.mean_yes_errors_all_results[threshold]))
            _, ax = plt.subplots()
            for i in range(len(self.thresholds)):
                ax.errorbar(
                    x[i], 
                    y[i], 
                    yerr=self.std_yes_errors_all_results[self.thresholds[i]], 
                    label=self.thresholds[i], 
                    errorevery=5
                    )
            ax.legend(loc="best")
            ax.set(xlabel=xlabel, ylabel='Error rate', 
            title=" 'Yes' action error rate - All results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # No errors all results
        with doc.create(Section(" 'No' action error rate - All results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for threshold in self.thresholds:
                x.append(np.array(range(1, len(self.mean_no_errors_all_results[threshold]) + 1)))
                y.append(np.array(self.mean_no_errors_all_results[threshold]))
            _, ax = plt.subplots()
            for i in range(len(self.thresholds)):
                ax.errorbar(
                    x[i], 
                    y[i], 
                    yerr=self.std_no_errors_all_results[self.thresholds[i]], 
                    label=self.thresholds[i], 
                    errorevery=5
                    )
            ax.legend(loc="best")
            ax.set(xlabel=xlabel, ylabel='Error rate', 
            title=" 'No' action error rate - All results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Numeric errors all results
        with doc.create(Section("Numeric action error rate - All results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for threshold in self.thresholds:
                x.append(np.array(range(1, len(self.mean_numeric_errors_all_results[threshold]) + 1)))
                y.append(np.array(self.mean_numeric_errors_all_results[threshold]))
            _, ax = plt.subplots()
            for i in range(len(self.thresholds)):
                ax.errorbar(
                    x[i], 
                    y[i], 
                    yerr=self.std_numeric_errors_all_results[self.thresholds[i]], 
                    label=self.thresholds[i], 
                    errorevery=5
                    )
            ax.legend(loc="best")
            ax.set(xlabel=xlabel, ylabel='Error rate', 
            title="Numeric action error rate - All results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Games against Glenn's heuristic
        with doc.create(Section("Games against Glenn's heuristic")):
            self.info_header_latex(doc)
            x_pos = np.arange(len(self.thresholds))
            vic = []
            draw = []
            loss = []
            error_vic = []
            error_draw = []
            error_loss = []
            for threshold in self.thresholds:
                vic.append(self.victories[threshold])
                draw.append(self.draws[threshold])
                loss.append(self.losses[threshold])
                error_vic.append(self.std_victories[threshold])
                error_draw.append(self.std_draws[threshold])
                error_loss.append(self.std_losses[threshold])
        
            X = np.arange(len(self.thresholds))
            plt.bar(X, vic, yerr=error_vic, color = 'g', width = 0.25, label='Victory')  
            plt.bar(X + 0.25, draw, yerr=error_draw, color = 'gray', width = 0.25, label='Draw')
            plt.bar(X + 0.5, loss, yerr=error_loss, color = 'r', width = 0.25, label='Loss')
            plt.xticks(X + .25, self.thresholds)
            plt.legend(labels=self.thresholds)
            plt.legend(loc="best")
            plt.ylabel('Number of games')
            plt.title(str(self.n_games_glenn) + " games against Glenn's heuristic")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        doc.generate_pdf(clean_tex=False)


    def batch_run(self):

        def avg(x):
            return sum(x) / len(x)

        def std(x):
            avg = sum(x) / len(x)
            aux = 0
            for value in x:
              aux += (value - avg)**2 
            return math.sqrt((1/len(x)) * aux)

        def slice(array):
            min_length = self.iterations
            for elem in array:
                if len(elem) < min_length:
                    min_length = len(elem)
            # Slice it
            array = [elem[:min_length] for elem in array]
            return array

    

        for i in range(len(self.thresholds)):

            full_results_total_errors_passed_results = []
            full_results_yes_errors_passed_results = []
            full_results_no_errors_passed_results = []
            full_results_numeric_errors_passed_results = []

            full_results_total_errors_all_results = []
            full_results_yes_errors_all_results = []
            full_results_no_errors_all_results = []
            full_results_numeric_errors_all_results = []

            batch_victories = []
            batch_draws = []
            batch_losses = []

            for j in range(self.batch_iterations):

                player_1 = Vanilla_UCT(c = 1, n_simulations = 500)
                player_2 = Vanilla_UCT(c = 1, n_simulations = 500)
                glenn = Rule_of_28_Player()

                if self.use_SA:
                    # optimization_algorithm = Metropolis-Hastings or 
                    #                          Simulated Annealing
                    opt_algo = SimulatedAnnealing(
                                                self.beta, 
                                                player_1, 
                                                player_2, 
                                                self.n_games, 
                                                self.iterations, 
                                                self.k,
                                                self.thresholds[i],
                                                self.tree_max_nodes,
                                                self.dataset_name,
                                                self.n_cores,
                                                self.d,
                                                self.init_temp
                                            )
                else:
                    opt_algo = MetropolisHastings(
                                                self.beta, 
                                                player_1, 
                                                player_2, 
                                                self.n_games, 
                                                self.iterations, 
                                                self.k,
                                                self.thresholds[i],
                                                self.tree_max_nodes,
                                                self.dataset_name,
                                                self.n_cores
                                            )

                best_program, script_best_player = opt_algo.run()
                
                script = Script(
                                    best_program, 
                                    self.k, 
                                    self.iterations, 
                                    self.tree_max_nodes
                                )
                sub_folder_batch = self.folder + '/batch_' + str(j) + '/'
                if not os.path.exists(sub_folder_batch):
                    os.makedirs(sub_folder_batch)
                script.save_file_custom(sub_folder_batch, 'threshold_' + str(self.thresholds[i]) + 'batch_' + str(j))

                one_iteration_total_errors_passed_results = []
                one_iteration_yes_errors_passed_results = []
                one_iteration_no_errors_passed_results = []
                one_iteration_numeric_errors_passed_results = []

                one_iteration_total_errors_all_results = []
                one_iteration_yes_errors_all_results = []
                one_iteration_no_errors_all_results = []
                one_iteration_numeric_errors_all_results = []

                # Error rate - Passed results
                for passed_data in opt_algo.passed_results:
                    one_iteration_total_errors_passed_results.append(passed_data[4])
                    one_iteration_yes_errors_passed_results.append(passed_data[5])
                    one_iteration_no_errors_passed_results.append(passed_data[6])
                    one_iteration_numeric_errors_passed_results.append(passed_data[7])
                # Error rate - All results
                for all_data in opt_algo.all_results:
                    one_iteration_total_errors_all_results.append(all_data[4])
                    one_iteration_yes_errors_all_results.append(all_data[5])
                    one_iteration_no_errors_all_results.append(all_data[6])
                    one_iteration_numeric_errors_all_results.append(all_data[7])

                full_results_total_errors_passed_results.append(one_iteration_total_errors_passed_results)
                full_results_yes_errors_passed_results.append(one_iteration_yes_errors_passed_results)
                full_results_no_errors_passed_results.append(one_iteration_no_errors_passed_results)
                full_results_numeric_errors_passed_results.append(one_iteration_numeric_errors_passed_results)

                full_results_total_errors_all_results.append(one_iteration_total_errors_all_results)
                full_results_yes_errors_all_results.append(one_iteration_yes_errors_all_results)
                full_results_no_errors_all_results.append(one_iteration_no_errors_all_results)
                full_results_numeric_errors_all_results.append(one_iteration_numeric_errors_all_results)

                # Play games against Glenn's heuristic for evaluation
                victories = 0
                losses = 0
                draws = 0
                for j in range(self.n_games_glenn):
                    game = Game(2, 4, 6, [2,12], 2, 2)
                    if j%2 == 0:
                        who_won = simplified_play_single_game(script_best_player, glenn, game, 500)
                        if who_won == 1:
                            victories += 1
                        elif who_won == 2:
                            losses += 1
                        else:
                            draws += 1
                    else:
                        who_won = simplified_play_single_game(glenn, script_best_player, game, 500)
                        if who_won == 2:
                            victories += 1
                        elif who_won == 1:
                            losses += 1
                        else:
                            draws += 1

                batch_victories.append(victories)
                batch_draws.append(draws)
                batch_losses.append(losses)


            self.data_distribution.append(opt_algo.data_distribution)
            # For "passed" results", some lists will be bigger than others because
            # every opt_algo (MH or SA) iteration will be different from others

            # Find out the minimum length of all runs, and slices all the 
            # remaining lists
            full_results_total_errors_passed_results = slice(full_results_total_errors_passed_results)
            full_results_yes_errors_passed_results = slice(full_results_yes_errors_passed_results)
            full_results_no_errors_passed_results = slice(full_results_no_errors_passed_results)
            full_results_numeric_errors_passed_results = slice(full_results_numeric_errors_passed_results)
            
            self.mean_total_errors_passed_results[self.thresholds[i]] = list(map(avg, zip_longest(*full_results_total_errors_passed_results)))
            self.mean_yes_errors_passed_results[self.thresholds[i]] = list(map(avg, zip_longest(*full_results_yes_errors_passed_results)))
            self.mean_no_errors_passed_results[self.thresholds[i]] = list(map(avg, zip_longest(*full_results_no_errors_passed_results)))
            self.mean_numeric_errors_passed_results[self.thresholds[i]] = list(map(avg, zip_longest(*full_results_numeric_errors_passed_results)))
            
            self.std_total_errors_passed_results[self.thresholds[i]] = list(map(std, zip_longest(*full_results_total_errors_passed_results)))
            self.std_yes_errors_passed_results[self.thresholds[i]] = list(map(std, zip_longest(*full_results_yes_errors_passed_results)))
            self.std_no_errors_passed_results[self.thresholds[i]] = list(map(std, zip_longest(*full_results_no_errors_passed_results)))
            self.std_numeric_errors_passed_results[self.thresholds[i]] = list(map(std, zip_longest(*full_results_numeric_errors_passed_results)))

            # "All results" don't need slicing
            self.mean_total_errors_all_results[self.thresholds[i]] = list(map(avg, zip_longest(*full_results_total_errors_all_results)))
            self.mean_yes_errors_all_results[self.thresholds[i]] = list(map(avg, zip_longest(*full_results_yes_errors_all_results)))
            self.mean_no_errors_all_results[self.thresholds[i]] = list(map(avg, zip_longest(*full_results_no_errors_all_results)))
            self.mean_numeric_errors_all_results[self.thresholds[i]] = list(map(avg, zip_longest(*full_results_numeric_errors_all_results)))

            self.std_total_errors_all_results[self.thresholds[i]] = list(map(std, zip_longest(*full_results_total_errors_all_results)))
            self.std_yes_errors_all_results[self.thresholds[i]] = list(map(std, zip_longest(*full_results_yes_errors_all_results)))
            self.std_no_errors_all_results[self.thresholds[i]] = list(map(std, zip_longest(*full_results_no_errors_all_results)))
            self.std_numeric_errors_all_results[self.thresholds[i]] = list(map(std, zip_longest(*full_results_numeric_errors_all_results)))

            # Glenn's games
            self.victories[self.thresholds[i]] = avg(batch_victories)
            self.draws[self.thresholds[i]] = avg(batch_draws)
            self.losses[self.thresholds[i]] = avg(batch_losses)
            self.std_victories[self.thresholds[i]] = std(batch_victories)
            self.std_draws[self.thresholds[i]] = std(batch_draws)
            self.std_losses[self.thresholds[i]] = std(batch_losses)

        self.generate_batch_report()

if __name__ == "__main__":
    beta = 0.5
    n_games = 750
    iterations = 2
    batch_iterations = 2
    k = -1
    thresholds = [0, 0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75]
    tree_max_nodes = 300
    n_cores = multiprocessing.cpu_count()
    use_SA = True
    d = 1
    init_temp = 1
    n_games_glenn = 100
    dataset_name = 'fulldata_sorted'

    experiment = ThresholdExperiment(
                                beta, n_games, iterations, batch_iterations, 
                                k, thresholds, tree_max_nodes, n_cores, use_SA, 
                                d, init_temp, n_games_glenn, dataset_name
                                )
    experiment.batch_run()
    
    

          

    