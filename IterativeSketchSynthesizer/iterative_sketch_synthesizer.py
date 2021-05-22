import time
import sys
import itertools
import os
import pickle
import random
from bottom_up_search import BottomUpSearch
from monte_carlo_simulations import MonteCarloSimulation
from simulated_annealing import SimulatedAnnealing
from DSL import *
sys.path.insert(0,'..')
from game import Game

class DSLTerms:
    def __init__(self, operations, constant_values, variables_list, variables_scalar_from_array, functions_scalars):
        self.operations = operations
        self.constant_values = constant_values
        self.variables_list = variables_list
        self.variables_scalar_from_array = variables_scalar_from_array
        self.functions_scalars = functions_scalars
        self.all_terms = []
        self.all_terms_in_order = []

    def get_all_terms(self):
        self.all_terms = [op.className() for op in self.operations] + \
                        [op.className() for op in self.constant_values] + \
                        [op.className() for op in self.variables_list] + \
                        [op.className() for op in self.variables_scalar_from_array] + \
                        [op.className() for op in self.functions_scalars]
        # No duplicates, maintain order
        return list(dict.fromkeys(self.all_terms))

    def add_term(self, full_DSL, to_be_added):
        self.all_terms_in_order.append(to_be_added)
        if to_be_added in ['Argmax', 'Sum', 'Map', 'Function', 'Plus', 'Times', 'Minus', 'Not']:
            to_be_added = [i for i in full_DSL.operations if i.className() == to_be_added]
            self.operations.append(to_be_added[0])
        elif to_be_added in ['Constant']:
            to_be_added = [i for i in full_DSL.constant_values if i.className() == to_be_added]
            self.constant_values.append(to_be_added[0])
        elif to_be_added in ['VarList', 'NoneNode']:
            to_be_added = [i for i in full_DSL.variables_list if i.className() == to_be_added]
            self.variables_list.append(to_be_added[0])
        elif to_be_added in ['VarScalarFromArray', 'VarScalar']:
            to_be_added = [i for i in full_DSL.variables_scalar_from_array if i.className() == to_be_added]
            self.variables_scalar_from_array.append(to_be_added[0])
        elif to_be_added in ['NumberAdvancedThisRound', 'NumberAdvancedByAction', 'IsNewNeutral', 'PlayerColumnAdvance', 'OpponentColumnAdvance', 'WillPlayerWinAfterN', 'AreThereAvailableColumnsToPlay']:
            to_be_added = [i for i in full_DSL.functions_scalars if i.className() == to_be_added]
            self.functions_scalars.append(to_be_added[0])
                      

class IterativeSketchSynthesizer:
    def __init__(self, MC_n_simulations, max_time, n_games, max_game_rounds, to_parallel, to_log, use_average, budget, n_run):
        self.MC_n_simulations = MC_n_simulations
        self.max_time = max_time
        self.n_games = n_games
        self.max_game_rounds = max_game_rounds
        self.to_parallel = to_parallel
        self.to_log = to_log
        self.use_average = use_average
        self.budget = budget
        self.n_run = n_run
        if self.use_average:
            self.folder = 'ISS_' + str(self.MC_n_simulations) + \
                            'sim_' + str(self.n_games) + 'games_use_average_' + \
                            str(self.n_run) + 'run/'
        else:
            self.folder = 'ISS_' + str(self.MC_n_simulations) + \
                            'sim_' + str(self.n_games) + 'games_use_best_' + \
                            str(self.n_run) + 'run/'
        if not os.path.exists(self.folder):
                os.makedirs(self.folder)

    def get_traversal(self, program):
        list_of_nodes = []
        self._get_traversal(program, list_of_nodes)
        return list_of_nodes

    def _get_traversal(self, program, list_of_nodes):
        list_of_nodes.append(program)
        for child in program.children:
            self._get_traversal(child, list_of_nodes)

    def update_parent(self, program, parent):
        program.add_parent(parent)
        for child in program.children:
            self.update_parent(child, program)

    def calculate_height(self, node):
        curr_height = 0
        while node.parent is not None:
            curr_height += 1
            node = node.parent
        return curr_height 


    def remove_terminal_nodes(self, terms_dict):
        to_delete = [
                        'Constant', 'VarList', 'NoneNode', 'VarScalarFromArray', 
                        'VarScalar', 'NumberAdvancedThisRound', 'NumberAdvancedByAction',
                        'IsNewNeutral', 'PlayerColumnAdvance', 'OpponentColumnAdvance',
                        'WillPlayerWinAfterN', 'AreThereAvailableColumnsToPlay', 'Not'
                    ]
        for term in to_delete:
            if term in terms_dict:
                del terms_dict[term]

    def run(self):
        start = time.time()
        
        partial_DSL = DSLTerms(
                                operations = [HoleNode], 
                                constant_values = [], 
                                variables_list = [], 
                                variables_scalar_from_array = [], 
                                functions_scalars = []
                            )
        
        full_DSL = DSLTerms(
                                operations = [Argmax, Sum, Map, Function, Plus, Times, Minus, Not], 
                                constant_values = [Constant(i) for i in range(20)], 
                                variables_list = [VarList('neutrals'), VarList('actions'), NoneNode()], 
                                variables_scalar_from_array = [VarScalarFromArray('progress_value'), VarScalarFromArray('move_value'), VarScalar('marker')], 
                                functions_scalars = [NumberAdvancedThisRound(), NumberAdvancedByAction(), IsNewNeutral(), PlayerColumnAdvance(), OpponentColumnAdvance(), WillPlayerWinAfterN(), AreThereAvailableColumnsToPlay()]
                            )

        partial_terms = partial_DSL.get_all_terms()
        all_terms = full_DSL.get_all_terms()

        #
        # Get all the sketches
        #

        all_closed_lists = []
        BUS = BottomUpSearch()
        for i in range(len(all_terms)):
            closed_list = BUS.synthesize(
                                    bound = 5, 
                                    partial_DSL = partial_DSL,
                                    folder = self.folder
                                )
            merged_closed_list = list(itertools.product(closed_list[1], repeat=2))
            # Stop the addition of DSL terms if the budget is reached
            if len(merged_closed_list) > self.budget:
                with open(self.folder + 'log_' + str(i) + '.txt', 'a') as f:
                    print('Closed list collected - Length before = ', len(closed_list[1]),  file=f)
                    print('Closed list collected - Length after  = ', len(merged_closed_list),  file=f)
                    print('Number of max sketches reached. Budget = ', self.budget, file=f)
                    print('Current partial DSL terms = ', partial_DSL.get_all_terms(), file=f)
                    print('The addition of the term', partial_DSL.get_all_terms()[-1], 'will  be ignored.', file=f)
                    break
            else:
                with open(self.folder + 'log_' + str(i) + '.txt', 'a') as f:
                    print('Closed list collected - Length before = ', len(closed_list[1]),  file=f)
                    print('Closed list collected - Length after  = ', len(merged_closed_list),  file=f)
                    print('Current partial DSL terms = ', partial_DSL.get_all_terms(), file=f)

            average = []
            victory = []
            start_MC = time.time()

            #
            # Evaluate all of them using Monte Carlo Simulations
            #

            # Dictionary to correlate a DSL term (key) to cummulative victory
            # and DSL term root proximity throughout the MC simulations (values)
            DSL_terms_scores = {}
            for term in all_terms:
                if term not in partial_DSL.get_all_terms():
                    DSL_terms_scores[term] = [0, []]
            for j, program in enumerate(merged_closed_list):
                start_MC_sim = time.time()
                MC = MonteCarloSimulation(program[0], program[1], self.MC_n_simulations, self.n_games, self.max_game_rounds, self.to_parallel, i, i, self.to_log)
                MC_data = MC.run()
                end_MC_sim = time.time() - start_MC_sim
                # Sorting by victory
                MC_data.sort_by_victory()
                with open(self.folder + 'log_' + str(i) + '.txt', 'a') as f:
                    print('Iteration -', j, '- Programs being simulated', file=f)
                    print(program[0].to_string(), ',', program[1].to_string(), file=f)
                    print('Avg V/L/D = ', MC_data.average_v, MC_data.average_l, MC_data.average_d, file=f)
                    print('Std V/L/D = ', MC_data.std_v, MC_data.std_l, MC_data.std_d, file=f)
                    print('Best programs simulated by MC', file=f)

                    # Stores the best (by number of victories) of each MC sim.
                    best_vic = []
                    for k in range(len(MC_data.simulation_victories[:3])):
                        best_vic.append(
                                        (
                                            program[0].to_string(), 
                                            program[1].to_string(), 
                                            MC_data.simulation_victories[k], 
                                            MC_data.simulation_losses[k], 
                                            MC_data.simulation_draws[k], 
                                            MC_data.programs_1_object[k], 
                                            MC_data.programs_2_object[k]
                                        )
                                    )
                        print(
                                'k = ', k, 
                                'V/L/D = ', \
                                MC_data.simulation_victories[k], \
                                MC_data.simulation_losses[k], \
                                MC_data.simulation_draws[k], \
                                'Programs = ', MC_data.programs_1_object[k].to_string(), \
                                ',', \
                                MC_data.programs_2_object[k].to_string(), \
                                file=f
                            )
                    print('Time elapsed of this MC simulation = ', end_MC_sim, file=f)
                    print(file=f)
                    victory.append(best_vic)
                
                average.append(
                                (
                                    program[0], 
                                    program[1],
                                    MC_data.average_v,
                                    MC_data.average_l,
                                    MC_data.average_d,
                                    MC_data.std_v,
                                    MC_data.std_l,
                                    MC_data.std_d
                                )
                            )
                # Calculate DSL terms victories and proximity to root
                for k in range(len(MC_data.programs_1_object)):
                    self.update_parent(MC_data.programs_1_object[k], None)
                    self.update_parent(MC_data.programs_2_object[k], None)
                    list_of_nodes_1 = self.get_traversal(MC_data.programs_1_object[k])
                    list_of_nodes_2 = self.get_traversal(MC_data.programs_2_object[k])
                    n_victory = MC_data.simulation_victories[k]
                    list_of_nodes_1_str = [node.className() for node in list_of_nodes_1]
                    list_of_nodes_2_str = [node.className() for node in list_of_nodes_2]
                    list_of_nodes_str_set = list(set(list_of_nodes_1_str + list_of_nodes_2_str))
                    # Add number of victories related to the DSL term
                    for node in list_of_nodes_str_set:
                        if node in DSL_terms_scores:
                            DSL_terms_scores[node][0] += n_victory
                    
                    # Calculate the distance from root of each node
                    list_of_nodes_str = list_of_nodes_1_str + list_of_nodes_2_str
                    list_of_nodes_obj = list_of_nodes_1 + list_of_nodes_2
                    distance = np.zeros(len(list_of_nodes_obj))
                    for m in range(len(list_of_nodes_obj)):
                        d = self.calculate_height(list_of_nodes_obj[m])
                        distance[m] = d

                    all_occurrences = list_of_nodes_1_str + list_of_nodes_2_str
                    for m in range(len(list_of_nodes_str_set)):
                        indices = [z for z, x in enumerate(all_occurrences) if x == list_of_nodes_str_set[m]]
                        ave_height = 0
                        for index in indices:
                            ave_height += distance[index]
                        ave_height = ave_height / len(indices)
                        if list_of_nodes_str_set[m] in DSL_terms_scores:
                            DSL_terms_scores[list_of_nodes_str_set[m]][1].append(ave_height)
            # Save best sketches (in terms of average) to file 
            average.sort(key=lambda tup: tup[2], reverse=True)
            with open(self.folder + 'average_log_' + str(i) + '.txt', 'a') as f:
                print('Average - len = ', len(average), file=f)
                for ave in average:
                    print('Sketch 1 and 2 = ', ave[0].to_string(), ave[1].to_string(), file=f)
                    print('Average V/L/D = ', ave[2], ave[3], ave[4], file=f)
                    print('Stdev V/L/D = ', ave[5], ave[6], ave[7], file=f)
                    print(file=f)
            # Save best programs (in terms of victory) to file 
            victory.sort(key=lambda tup: tup[0][2], reverse=True)
            with open(self.folder + 'victory_log_' + str(i) + '.txt', 'a') as f:
                print('Victory - len = ', len(victory), file=f)
                for vic_best in victory:
                    print('Sketch = ', vic_best[0][0], vic_best[0][1], file=f)
                    for vic in vic_best:
                        print('V/L/D = ', vic[2], vic[3], vic[4], vic[5].to_string(), vic[6].to_string(), file=f)
                    print(file=f)
            # Save both victory and average to file
            with open(self.folder + 'average' + str(i), 'wb') as file:
                pickle.dump(average, file)
            with open(self.folder + 'victory' + str(i), 'wb') as file:
                pickle.dump(victory, file)
            end_MC = time.time() - start_MC
            with open(self.folder + 'log_' + str(i) + '.txt', 'a') as f:
                print('MC Time elapsed = ', end_MC, file=f)

            # Remove terminal DSL terms from the dictionary
            self.remove_terminal_nodes(DSL_terms_scores)
            with open(self.folder + 'log_' + str(i) + '.txt', 'a') as f:
                print('DSL dict. after MC', file=f)
                print(DSL_terms_scores, file=f)
                print('DSL dict. after MC (formatted)', file=f)
                for k in DSL_terms_scores:
                    if len(DSL_terms_scores[k][1]) != 0:
                        print('Term = ', k, 'Vic = ', DSL_terms_scores[k][0], 'Ave. height = ', sum(DSL_terms_scores[k][1])/len(DSL_terms_scores[k][1]), file=f)

            # Auxiliar function to get the average while giving a low score
            # if list is empty (term not used in the simulations)
            def calc_ave(l):
                if len(l) == 0:
                    return -99999999
                else:
                    return sum(l)/len(l)
            # Get the DSL term with the highest number of victories.
            # If tied: take the one closest to the root (in average).
            # If tied: pick a term randomly.
            term_chosen = max(DSL_terms_scores, key=lambda key_term: (DSL_terms_scores[key_term][0], -calc_ave(DSL_terms_scores[key_term][1]), random.random()))
            with open(self.folder + 'log_' + str(i) + '.txt', 'a') as f:
                print('DSL term chosen = ', term_chosen, file=f)
            partial_DSL.add_term(full_DSL, term_chosen)

        #
        # Run Simulated Annealing on the best ones
        #

        best_average_1 = average[0][0]
        best_average_2 = average[0][1]

        best_victory_1 = victory[0][0][5]
        best_victory_2 = victory[0][0][6]

        path_pre_sa = self.folder + 'log_pre_sa.txt'
        with open(path_pre_sa, 'a') as f:
            print('best_average_1', file=f)
            print(best_average_1.to_string(), file=f)
        best_average_1.print_tree_file(path_pre_sa)
        with open(path_pre_sa, 'a') as f:
            print('best_average_2', file=f)
            print(best_average_2.to_string(), file=f)
        best_average_2.print_tree_file(path_pre_sa)
        with open(path_pre_sa, 'a') as f:
            print('best_victory_1', file=f)
            print(best_victory_1.to_string(), file=f)
        best_victory_1.print_tree_file(path_pre_sa)
        with open(path_pre_sa, 'a') as f:
            print('best_victory_2', file=f)
            print(best_victory_2.to_string(), file=f)
        best_victory_2.print_tree_file(path_pre_sa)

        n_SA_iterations = 200000
        max_game_rounds = 500
        n_games = 1000
        init_temp = 1
        d = 1
        algo_name = self.folder
        initial_time = time.time() - start
        start_SA = time.time()
        SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name, initial_time, self.max_time)
        if self.use_average:
            _, _, _, _, _, _  = SA.run(best_average_1, best_average_2, False)
        else:
            _, _, _, _, _, _  = SA.run(best_victory_1, best_victory_2, True)
        end_SA = time.time() - start_SA
        print('Time elapsed = ', end_SA)

        end = time.time()
        print('Running Time: ', end - start, ' seconds.')
        return all_closed_lists

if __name__ == "__main__":
    MC_n_simulations = 5
    # Stop condition for SA (in seconds)
    max_time = 360
    n_games = 100
    max_game_rounds = 500
    to_parallel = False
    to_log = False
    use_average = False
    budget = 2000
    n_run = 6
    start_ISS = time.time()
    ISS = IterativeSketchSynthesizer(MC_n_simulations, max_time, n_games, max_game_rounds, to_parallel, to_log, use_average, budget, n_run)
    all_closed_lists = ISS.run()
    end_ISS = time.time() - start_ISS
    print('ISS - Time elapsed = ', end_ISS)