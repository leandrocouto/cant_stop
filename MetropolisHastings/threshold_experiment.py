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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from pylatex import Document, Section, Figure, NoEscape
from pylatex.utils import bold
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def info_header_latex(doc, data_distribution, thresholds):
    """Create a header used before each graph in the report."""

    doc.append(bold('Action distribution for different thresholds: '))
    #doc.append(str(self.n_simulations))
    for i in range(len(thresholds)):
        n_actions = data_distribution[i][0] + data_distribution[i][1] + data_distribution[i][2]
        yes_percentage = round((data_distribution[i][0] / n_actions) * 100, 2)
        no_percentage = round((data_distribution[i][1] / n_actions) * 100, 2)
        num_percentage = round((data_distribution[i][2] / n_actions) * 100, 2)
        doc.append(bold('\nThreshold = '))
        doc.append(str(thresholds[i]))
        doc.append(bold(' Total actions = '))
        doc.append(str(n_actions))
        doc.append(bold("  'Yes' = "))
        doc.append(str(data_distribution[i][0]))
        doc.append("  (")
        doc.append(str(yes_percentage))
        doc.append("%)")
        doc.append(bold("  'No' = "))
        doc.append(str(data_distribution[i][1]))
        doc.append("  (")
        doc.append(str(no_percentage))
        doc.append("%)")
        doc.append(bold("  Num = "))
        doc.append(str(data_distribution[i][2]))
        doc.append("  (")
        doc.append(str(num_percentage))
        doc.append("%)")

def generate_report(total_errors_passed_results, yes_errors_passed_results,
    no_errors_passed_results, numeric_errors_passed_results,
    total_errors_all_results, yes_errors_all_results, no_errors_all_results,
    numeric_errors_all_results, data_distribution, thresholds):

    matplotlib.use('Agg') 

    geometry_options = {"right": "2cm", "left": "2cm"}
    #pdf_name = str(self.n_simulations) + '_' + str(self.n_games) \
    #        + '_' + str(self.alphazero_iterations) + '_' \
    #        + str(self.conv_number) + '_' + str(self.use_UCT_playout)
    
    pdf_name = "testando"
    doc = Document(pdf_name, geometry_options=geometry_options)

    # Total error passed results
    with doc.create(Section('Total error rate - Passed results')):
        info_header_latex(doc, data_distribution, thresholds)
        x = []
        y = []
        for i in range(len(total_errors_passed_results)):
            x.append(np.array(range(1, len(total_errors_passed_results[i]) + 1)))
            y.append(np.array(total_errors_passed_results[i]))
        _, ax = plt.subplots()
        for i in range(len(total_errors_passed_results)):
            ax.plot(x[i], y[i], label=thresholds[i])
        ax.legend(loc="best")
        ax.set(xlabel='MH Iterations', ylabel='Error rate', 
        title='Error rate - Passed results')

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

    plt.close()
    doc.append(NoEscape(r'\newpage'))

    # Yes errors passed results
    with doc.create(Section(" 'Yes' action error rate - Passed results")):
        info_header_latex(doc, data_distribution, thresholds)
        x = []
        y = []
        for i in range(len(yes_errors_passed_results)):
            x.append(np.array(range(1, len(yes_errors_passed_results[i]) + 1)))
            y.append(np.array(yes_errors_passed_results[i]))

        _, ax = plt.subplots()
        for i in range(len(yes_errors_passed_results)):
            ax.plot(x[i], y[i], label=thresholds[i])
        ax.legend(loc="best")
        ax.set(xlabel='MH Iterations', ylabel='Error rate', 
        title="'Yes' action error rate - Passed results")

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

    # No errors passed results
    with doc.create(Section(" 'No' action error rate - Passed results")):
        info_header_latex(doc, data_distribution, thresholds)
        x = []
        y = []
        for i in range(len(no_errors_passed_results)):
            x.append(np.array(range(1, len(no_errors_passed_results[i]) + 1)))
            y.append(np.array(no_errors_passed_results[i]))

        _, ax = plt.subplots()
        for i in range(len(no_errors_passed_results)):
            ax.plot(x[i], y[i], label=thresholds[i])
        ax.legend(loc="best")
        ax.set(xlabel='MH Iterations', ylabel='Error rate', 
        title="'No' action error rate - Passed results")

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

    # Numeric errors passed results
    with doc.create(Section('Numeric error rate - Passed results')):
        info_header_latex(doc, data_distribution, thresholds)
        x = []
        y = []
        for i in range(len(numeric_errors_passed_results)):
            x.append(np.array(range(1, len(numeric_errors_passed_results[i]) + 1)))
            y.append(np.array(numeric_errors_passed_results[i]))

        _, ax = plt.subplots()
        for i in range(len(numeric_errors_passed_results)):
            ax.plot(x[i], y[i], label=thresholds[i])
        ax.legend(loc="best")
        ax.set(xlabel='MH Iterations', ylabel='Error rate', 
        title='Numeric error rate - Passed results')

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))


    # Total error all results
    with doc.create(Section('Error rate - All results')):
        info_header_latex(doc, data_distribution, thresholds)
        x = []
        y = []
        for i in range(len(total_errors_all_results)):
            x.append(np.array(range(1, len(total_errors_all_results[i]) + 1)))
            y.append(np.array(total_errors_all_results[i]))
        _, ax = plt.subplots()
        for i in range(len(total_errors_all_results)):
            ax.plot(x[i], y[i], label=thresholds[i])
        ax.legend(loc="best")
        ax.set(xlabel='MH Iterations', ylabel='Error rate', 
        title='Error rate - All results')

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

    plt.close()
    doc.append(NoEscape(r'\newpage'))

    # 'Yes' errors all results
    with doc.create(Section("'Yes' action error rate - All results")):
        info_header_latex(doc, data_distribution, thresholds)
        x = []
        y = []
        for i in range(len(yes_errors_all_results)):
            x.append(np.array(range(1, len(yes_errors_all_results[i]) + 1)))
            y.append(np.array(yes_errors_all_results[i]))

        _, ax = plt.subplots()
        for i in range(len(yes_errors_all_results)):
            ax.plot(x[i], y[i], label=thresholds[i])
        ax.legend(loc="best")
        ax.set(xlabel='MH Iterations', ylabel='Error rate', 
        title="'Yes' action error rate - All results")

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

    # 'No' errors all results
    with doc.create(Section("'No' action error rate - All results")):
        info_header_latex(doc, data_distribution, thresholds)
        x = []
        y = []
        for i in range(len(no_errors_all_results)):
            x.append(np.array(range(1, len(no_errors_all_results[i]) + 1)))
            y.append(np.array(no_errors_all_results[i]))

        _, ax = plt.subplots()
        for i in range(len(no_errors_all_results)):
            ax.plot(x[i], y[i], label=thresholds[i])
        ax.legend(loc="best")
        ax.set(xlabel='MH Iterations', ylabel='Error rate', 
        title="'No' action error rate - All results")

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

    # Numeric errors all results
    with doc.create(Section('Numeric error rate - All results')):
        info_header_latex(doc, data_distribution, thresholds)
        x = []
        y = []
        for i in range(len(numeric_errors_all_results)):
            x.append(np.array(range(1, len(numeric_errors_all_results[i]) + 1)))
            y.append(np.array(numeric_errors_all_results[i]))

        _, ax = plt.subplots()
        for i in range(len(numeric_errors_all_results)):
            ax.plot(x[i], y[i], label=thresholds[i])
        ax.legend(loc="best")
        ax.set(xlabel='MH Iterations', ylabel='Error rate', 
        title='Numeric error rate - All results')

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

    # 100 games against Glenn's Rule of 28 heuristic
    with doc.create(Section("Victories of generated script against Glenn's heuristic")):
        info_header_latex(doc, data_distribution, thresholds)

        with doc.create(Figure(position='h!')) as vic:
            vic.add_image('victories.png')#, width='120px')
        '''
        games_against_rule_of_28 = [30, 50]
        #_, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        print('thresholds = ', thresholds)
        print('games_against_rule_of_28 = ', games_against_rule_of_28)
        ax.bar(thresholds, games_against_rule_of_28)
        ax.legend(labels=thresholds)
        #ax.legend(loc="best")

        #plt.show()

        with doc.create(Figure(position='htbp')) as plot:
            plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)
        '''
        
        plt.close()
        doc.append(NoEscape(r'\newpage'))

    doc.generate_pdf(clean_tex=False)

if __name__ == "__main__":
    player_1 = Vanilla_UCT(c = 1, n_simulations = 500)
    player_2 = Vanilla_UCT(c = 1, n_simulations = 500)
    beta = 0.5
    n_games = 750
    iterations = 500
    k = -1
    thresholds = [0, 0.25, 0.50, 0.75, 1, 1.25, 1.50, 1.75]
    tree_max_nodes = 300
    n_cores = multiprocessing.cpu_count()
    # Simulated annealing parameters
    # If no SA is to be used, set both parameters to 1.
    temperature = 1
    temperature_dec = 1

    total_errors_passed_results = []
    yes_errors_passed_results = []
    no_errors_passed_results = []
    numeric_errors_passed_results = []

    total_errors_all_results = []
    yes_errors_all_results = []
    no_errors_all_results = []
    numeric_errors_all_results = []

    data_distribution = []

    victories_rule_of_28 = []
    losses_rule_of_28 = []
    draws_rule_of_28 = []

    for i in range(len(thresholds)):

        MH = MetropolisHastings(
                                    beta, 
                                    player_1, 
                                    player_2, 
                                    n_games, 
                                    iterations, 
                                    k,
                                    thresholds[i],
                                    tree_max_nodes,
                                    temperature,
                                    temperature_dec,
                                    'fulldata_sorted',
                                    n_cores
                                )

        best_program, script_best_player = MH.run()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        script = Script(
                            best_program, 
                            k, 
                            iterations, 
                            tree_max_nodes
                        )
        script.save_file_custom(dir_path, 'threshold_' + str(thresholds[i]))

        glenn = Rule_of_28_Player()

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

        victories_rule_of_28.append(victories)
        losses_rule_of_28.append(losses)
        draws_rule_of_28.append(draws)


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

        total_errors_passed_results.append(one_iteration_total_errors_passed_results)
        yes_errors_passed_results.append(one_iteration_yes_errors_passed_results)
        no_errors_passed_results.append(one_iteration_no_errors_passed_results)
        numeric_errors_passed_results.append(one_iteration_numeric_errors_passed_results)

        total_errors_all_results.append(one_iteration_total_errors_all_results)
        yes_errors_all_results.append(one_iteration_yes_errors_all_results)
        no_errors_all_results.append(one_iteration_no_errors_all_results)
        numeric_errors_all_results.append(one_iteration_numeric_errors_all_results)

        data_distribution.append(MH.data_distribution)

    
    X = np.arange(len(thresholds))
    plt.bar(X, victories_rule_of_28, color = 'b', width = 0.25, label='Victory')
    plt.bar(X + 0.25, losses_rule_of_28, color = 'g', width = 0.25, label='Loss')
    plt.bar(X + 0.5, draws_rule_of_28, color = 'r', width = 0.25, label='Draw')
    plt.xticks(X + .25, thresholds)
    plt.legend(labels=thresholds)
    plt.legend(loc="best")
    plt.ylabel('Victories')
    plt.title("Victories by synthesized scripts against Glenn's heuristic")
    plt.savefig('victories.png')
    generate_report(
                    total_errors_passed_results,
                    yes_errors_passed_results,
                    no_errors_passed_results,
                    numeric_errors_passed_results,
                    total_errors_all_results,
                    yes_errors_all_results,
                    no_errors_all_results,
                    numeric_errors_all_results,
                    data_distribution,
                    thresholds
                    )

          

    