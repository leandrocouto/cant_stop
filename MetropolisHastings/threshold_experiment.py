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
from play_game_template import simplified_play_single_game
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
        self.iterations = 5
        self.batch_iterations = 3
        self.k = -1
        self.thresholds = [0, 0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75]
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

    def generate_report(self):

        matplotlib.use('Agg') 

        geometry_options = {"right": "2cm", "left": "2cm"}
        #pdf_name = str(self.n_simulations) + '_' + str(self.n_games) \
        #        + '_' + str(self.alphazero_iterations) + '_' \
        #        + str(self.conv_number) + '_' + str(self.use_UCT_playout)
        
        pdf_name = "testando"
        doc = Document(pdf_name, geometry_options=geometry_options)

        # Total error passed results
        with doc.create(Section('Total error rate - Passed results')):
            self.info_header_latex(doc)
            x = []
            y = []
            for i in range(len(self.total_errors_passed_results)):
                x.append(np.array(range(1, len(self.total_errors_passed_results[i]) + 1)))
                y.append(np.array(self.total_errors_passed_results[i]))
            _, ax = plt.subplots()
            for i in range(len(self.total_errors_passed_results)):
                ax.plot(x[i], y[i], label=self.thresholds[i])
            ax.legend(loc="best")
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title='Error rate - Passed results')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Yes errors passed results
        with doc.create(Section(" 'Yes' action error rate - Passed results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for i in range(len(self.yes_errors_passed_results)):
                x.append(np.array(range(1, len(self.yes_errors_passed_results[i]) + 1)))
                y.append(np.array(self.yes_errors_passed_results[i]))

            _, ax = plt.subplots()
            for i in range(len(self.yes_errors_passed_results)):
                ax.plot(x[i], y[i], label=self.thresholds[i])
            ax.legend(loc="best")
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title="'Yes' action error rate - Passed results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # No errors passed results
        with doc.create(Section(" 'No' action error rate - Passed results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for i in range(len(self.no_errors_passed_results)):
                x.append(np.array(range(1, len(self.no_errors_passed_results[i]) + 1)))
                y.append(np.array(self.no_errors_passed_results[i]))

            _, ax = plt.subplots()
            for i in range(len(self.no_errors_passed_results)):
                ax.plot(x[i], y[i], label=self.thresholds[i])
            ax.legend(loc="best")
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title="'No' action error rate - Passed results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # Numeric errors passed results
        with doc.create(Section('Numeric error rate - Passed results')):
            self.info_header_latex(doc)
            x = []
            y = []
            for i in range(len(self.numeric_errors_passed_results)):
                x.append(np.array(range(1, len(self.numeric_errors_passed_results[i]) + 1)))
                y.append(np.array(self.numeric_errors_passed_results[i]))

            _, ax = plt.subplots()
            for i in range(len(self.numeric_errors_passed_results)):
                ax.plot(x[i], y[i], label=self.thresholds[i])
            ax.legend(loc="best")
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title='Numeric error rate - Passed results')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))


        # Total error all results
        with doc.create(Section('Error rate - All results')):
            self.info_header_latex(doc)
            x = []
            y = []
            for i in range(len(self.total_errors_all_results)):
                x.append(np.array(range(1, len(self.total_errors_all_results[i]) + 1)))
                y.append(np.array(self.total_errors_all_results[i]))
            _, ax = plt.subplots()
            for i in range(len(self.total_errors_all_results)):
                ax.plot(x[i], y[i], label=self.thresholds[i])
            ax.legend(loc="best")
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title='Error rate - All results')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # 'Yes' errors all results
        with doc.create(Section("'Yes' action error rate - All results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for i in range(len(self.yes_errors_all_results)):
                x.append(np.array(range(1, len(self.yes_errors_all_results[i]) + 1)))
                y.append(np.array(self.yes_errors_all_results[i]))

            _, ax = plt.subplots()
            for i in range(len(self.yes_errors_all_results)):
                ax.plot(x[i], y[i], label=self.thresholds[i])
            ax.legend(loc="best")
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title="'Yes' action error rate - All results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # 'No' errors all results
        with doc.create(Section("'No' action error rate - All results")):
            self.info_header_latex(doc)
            x = []
            y = []
            for i in range(len(self.no_errors_all_results)):
                x.append(np.array(range(1, len(self.no_errors_all_results[i]) + 1)))
                y.append(np.array(self.no_errors_all_results[i]))

            _, ax = plt.subplots()
            for i in range(len(self.no_errors_all_results)):
                ax.plot(x[i], y[i], label=self.thresholds[i])
            ax.legend(loc="best")
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title="'No' action error rate - All results")

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # Numeric errors all results
        with doc.create(Section('Numeric error rate - All results')):
            self.info_header_latex(doc)
            x = []
            y = []
            for i in range(len(self.numeric_errors_all_results)):
                x.append(np.array(range(1, len(self.numeric_errors_all_results[i]) + 1)))
                y.append(np.array(self.numeric_errors_all_results[i]))

            _, ax = plt.subplots()
            for i in range(len(self.numeric_errors_all_results)):
                ax.plot(x[i], y[i], label=self.thresholds[i])
            ax.legend(loc="best")
            ax.set(xlabel='MH Iterations', ylabel='Error rate', 
            title='Numeric error rate - All results')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # 100 games against Glenn's Rule of 28 heuristic
        with doc.create(Section("Victories of generated script against Glenn's heuristic")):
            self.info_header_latex(doc)

            with doc.create(Figure(position='h!')) as vic:
                vic.add_image('victories.png')
            
            plt.close()
            doc.append(NoEscape(r'\newpage'))

        doc.generate_pdf(clean_tex=False)

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

    def run(self):

        for i in range(len(self.thresholds)):

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
            script.save_file_custom(dir_path, 'threshold_' + str(self.thresholds[i]))

            

            victories = 0
            losses = 0
            draws = 0
            for i in range(100):
                game = Game(2, 4, 6, [2,12], 2, 2)
                if i%2 == 0:
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

            self.victories_rule_of_28.append(victories)
            self.losses_rule_of_28.append(losses)
            self.draws_rule_of_28.append(draws)


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

            self.total_errors_passed_results.append(one_iteration_total_errors_passed_results)
            self.yes_errors_passed_results.append(one_iteration_yes_errors_passed_results)
            self.no_errors_passed_results.append(one_iteration_no_errors_passed_results)
            self.numeric_errors_passed_results.append(one_iteration_numeric_errors_passed_results)

            self.total_errors_all_results.append(one_iteration_total_errors_all_results)
            self.yes_errors_all_results.append(one_iteration_yes_errors_all_results)
            self.no_errors_all_results.append(one_iteration_no_errors_all_results)
            self.numeric_errors_all_results.append(one_iteration_numeric_errors_all_results)

            self.data_distribution.append(MH.data_distribution)

        X = np.arange(len(self.thresholds))
        plt.bar(X, self.victories_rule_of_28, color = 'b', width = 0.25, label='Victory')
        plt.bar(X + 0.25, self.losses_rule_of_28, color = 'g', width = 0.25, label='Loss')
        plt.bar(X + 0.5, self.draws_rule_of_28, color = 'r', width = 0.25, label='Draw')
        plt.xticks(X + .25, self.thresholds)
        plt.legend(labels=self.thresholds)
        plt.legend(loc="best")
        plt.ylabel('Victories')
        plt.title("Victories by synthesized scripts against Glenn's heuristic")
        plt.savefig('victories.png')
        self.generate_report()

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

                #print('colocando o resultado')
                #print(one_iteration_total_errors_passed_results)
                full_results_total_errors_passed_results.append(one_iteration_total_errors_passed_results)
                full_results_yes_errors_passed_results.append(one_iteration_yes_errors_passed_results)
                full_results_no_errors_passed_results.append(one_iteration_no_errors_passed_results)
                full_results_numeric_errors_passed_results.append(one_iteration_numeric_errors_passed_results)

                full_results_total_errors_all_results.append(one_iteration_total_errors_all_results)
                full_results_yes_errors_all_results.append(one_iteration_yes_errors_all_results)
                full_results_no_errors_all_results.append(one_iteration_no_errors_all_results)
                full_results_numeric_errors_all_results.append(one_iteration_numeric_errors_all_results)

                #print('data_distribution = ', MH.data_distribution)

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

        print('data_distribution')
        print(self.data_distribution)
        #exit()
        self.generate_batch_report()

if __name__ == "__main__":
    experiment = ThresholdExperiment()
    #experiment.run()
    experiment.batch_run()

    
    
    

          

    