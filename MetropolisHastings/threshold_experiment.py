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
    def __init__(self):
        self.beta = 0.5
        self.n_games = 750
        self.iterations = 300
        self.batch_iterations = 1
        self.k = -1
        self.thresholds = [0]#[0, 0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75]
        self.tree_max_nodes = 300
        self.n_cores = multiprocessing.cpu_count()
        self.temperature = 1
        self.temperature_dec = 1

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

        
        self.data_distribution = []
        self.victories_rule_of_28 = []
        self.losses_rule_of_28 = []
        self.draws_rule_of_28 = []

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
        
        pdf_name = "testando"
        doc = Document(pdf_name, geometry_options=geometry_options)

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
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
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
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
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
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
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
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
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
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
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
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
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
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
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
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title="Numeric action error rate - All results")

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

            for j in range(self.batch_iterations):

                player_1 = Vanilla_UCT(c = 1, n_simulations = 500)
                player_2 = Vanilla_UCT(c = 1, n_simulations = 500)
                glenn = Rule_of_28_Player()

                MH = MetropolisHastings(
                                            self.beta, 
                                            player_1, 
                                            player_2, 
                                            self.n_games, 
                                            self.iterations, 
                                            self.k,
                                            self.thresholds[i],
                                            self.tree_max_nodes,
                                            self.temperature,
                                            self.temperature_dec,
                                            'fulldata_sorted',
                                            self.n_cores
                                        )

                best_program, script_best_player = MH.run()

                dir_path = os.path.dirname(os.path.realpath(__file__))
                script = Script(
                                    best_program, 
                                    self.k, 
                                    self.iterations, 
                                    self.tree_max_nodes
                                )
                script.save_file_custom(dir_path, 'threshold_' + str(self.thresholds[i]) + '_batch_' + str(j))

                one_iteration_total_errors_passed_results = []
                one_iteration_yes_errors_passed_results = []
                one_iteration_no_errors_passed_results = []
                one_iteration_numeric_errors_passed_results = []

                one_iteration_total_errors_all_results = []
                one_iteration_yes_errors_all_results = []
                one_iteration_no_errors_all_results = []
                one_iteration_numeric_errors_all_results = []

                # Error rate - Passed results
                for passed_data in MH.passed_results:
                    one_iteration_total_errors_passed_results.append(passed_data[4])
                    one_iteration_yes_errors_passed_results.append(passed_data[5])
                    one_iteration_no_errors_passed_results.append(passed_data[6])
                    one_iteration_numeric_errors_passed_results.append(passed_data[7])
                # Error rate - All results
                for all_data in MH.all_results:
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


            self.data_distribution.append(MH.data_distribution)
            # For "passed" results", some lists will be bigger than others because
            # every MH iteration will be different from others

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

        self.generate_batch_report()

if __name__ == "__main__":
    experiment = ThresholdExperiment()
    #experiment.run()
    experiment.batch_run()
    '''
    draw = 0
    glenn_vic = 0
    uct_vic = 0
    n_games = 100
    n_simulations = 500

    glenn = Rule_of_28_Player()
    uct = Vanilla_UCT(c = 1, n_simulations = n_simulations)

    for i in range(n_games):
        game = Game(2, 4, 6, [2,12], 2, 2)
        if i%2 == 0:
            who_won = play_single_game(uct, glenn, game, 500)
            if who_won == 1:
                uct_vic += 1
                print(i, '- uct won')
            elif who_won == 2:
                glenn_vic += 1
                print(i, '- glenn won')
            else:
                draw += 1
                print(i, '- draw')
        else:
            who_won = play_single_game(glenn, uct, game, 500)
            if who_won == 1:
                print(i, '- glenn won')
                glenn_vic += 1
            elif who_won == 2:
                uct_vic += 1
                print(i, '- uct won')
            else:
                draw += 1
                print(i, '- draw')

    print('playouts = ', n_simulations)
    print('draw = ', draw)
    print('glenn_vic = ', glenn_vic)
    print('uct_vic = ', uct_vic)
    '''
    '''
    glenn = Rule_of_28_Player()
    new_data = []

    n_errors = 0
    n_errors_yes_action = 0
    n_errors_no_action = 0
    n_errors_numeric_action = 0

    n_data_yes_action = 0
    n_data_no_action = 0
    n_data_numeric_action = 0

    # Read the dataset
    with open('fulldata_sorted', "rb") as f:
        while True:
            try:
                new_data.append(pickle.load(f))
            except EOFError:
                break

    for i in range(len(new_data)):
        chosen_play = glenn.get_action(new_data[i][0])
        oracle_play = new_data[i][1]
        # Compare the action chosen by the synthesized script and the oracle
        if chosen_play != oracle_play:
            n_errors += 1

            if oracle_play == 'y':
                n_errors_yes_action += 1
            elif oracle_play == 'n':
                n_errors_no_action += 1
            else:
                n_errors_numeric_action += 1

        #For report purposes
        if oracle_play == 'y':
            n_data_yes_action += 1
        elif oracle_play == 'n':
            n_data_no_action += 1
        else:
            n_data_numeric_action += 1

    print('n_errors = ', n_errors, '(', round((n_errors/(n_data_yes_action + n_data_no_action + n_data_numeric_action))*100, 2), '%)')
    print('n_errors_yes_action = ', n_errors_yes_action, '(', round((n_errors_yes_action/n_data_yes_action)*100, 2), '%)')
    print('n_errors_no_action = ', n_errors_no_action, '(', round((n_errors_no_action/n_data_no_action)*100, 2), '%)')
    print('n_errors_numeric_action = ', n_errors_numeric_action, '(', round((n_errors_numeric_action/n_data_numeric_action)*100, 2), '%)')
    print('n_data_yes_action = ', n_data_yes_action)
    print('n_data_no_action = ', n_data_no_action)
    print('n_data_numeric_action = ', n_data_numeric_action)
    '''


    
    
    

          

    