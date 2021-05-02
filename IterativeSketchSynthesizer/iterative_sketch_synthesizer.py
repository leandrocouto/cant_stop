import time
import sys
import itertools
import os
import pickle
from bottom_up_search import BottomUpSearch
from monte_carlo_simulations import MonteCarloSimulation
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

    def get_all_terms(self):
        self.all_terms = [op.className() for op in self.operations] + \
                        [op.className() for op in self.constant_values] + \
                        [op.className() for op in self.variables_list] + \
                        [op.className() for op in self.variables_scalar_from_array] + \
                        [op.className() for op in self.functions_scalars]
        # No duplicates, maintain order
        return list(dict.fromkeys(self.all_terms))

class IterativeSketchSynthesizer:
    def __init__(self, n_terms, MC_n_simulations, n_games, max_game_rounds, to_parallel, to_log):
        self.n_terms = n_terms
        self.MC_n_simulations = MC_n_simulations
        self.n_games = n_games
        self.max_game_rounds = max_game_rounds
        self.to_parallel = to_parallel
        self.to_log = to_log
        self.folder = 'ISS_' + str(self.n_terms) + 'terms_' + str(self.MC_n_simulations) + \
                        'sim_' + str(self.n_games) + 'games/'
        if not os.path.exists(self.folder):
                os.makedirs(self.folder)

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
                                operations = [Argmax, Sum, Map, Function, Plus, Times, Minus], 
                                constant_values = [], 
                                variables_list = [VarList('neutrals'), VarList('actions')], 
                                variables_scalar_from_array = [VarScalarFromArray('progress_value'), VarScalarFromArray('move_value')], 
                                functions_scalars = [NumberAdvancedThisRound(), NumberAdvancedByAction(), IsNewNeutral(), PlayerColumnAdvance(), OpponentColumnAdvance()]
                            )

        all_terms = full_DSL.get_all_terms()

        #
        # Get all the sketches
        #

        BUS = BottomUpSearch()
        all_closed_lists = BUS.synthesize(
                                bound = 10, 
                                partial_DSL = partial_DSL,
                                full_DSL = full_DSL,
                                n_terms = self.n_terms,
                                folder = self.folder
                            )
        closed_list = all_closed_lists[-1][1]
        merged_closed_list = list(itertools.product(closed_list, repeat=2))
        with open(self.folder + 'log.txt', 'a') as f:
            print('Closed list collected - Length before = ', len(closed_list),  file=f)
            print('Closed list collected - Length after  = ', len(merged_closed_list),  file=f)

        average = []
        all_MC_data = []
        start_MC = time.time()

        #
        # Evaluate all of them using Monte Carlo Simulations
        #

        for i, program in enumerate(merged_closed_list):
            MC = MonteCarloSimulation(program[0], program[1], self.MC_n_simulations, self.n_games, self.max_game_rounds, self.to_parallel, i, i, self.to_log)
            MC_data = MC.run()
            # Sorting by victory
            MC_data.sort_by_victory()
            with open(self.folder + 'log.txt', 'a') as f:
                print('Iteration -', i, '- Programs being simulated', file=f)
                print(program[0].to_string(), ',', program[1].to_string(), file=f)
                print('Avg V/L/D = ', MC_data.average_v, MC_data.average_l, MC_data.average_d, file=f)
                print('Std V/L/D = ', MC_data.std_v, MC_data.std_l, MC_data.std_d, file=f)
                print('Best 3 programs simulated by MC', file=f)
                for i in range(len(MC_data.simulation_victories[:3])):
                    print(
                            'i = ', i, 
                            'V/L/D = ', \
                            MC_data.simulation_victories[i], \
                            MC_data.simulation_losses[i], \
                            MC_data.simulation_draws[i], \
                            'Programs = ', MC_data.programs_1_object[i].to_string(), \
                            ',', \
                            MC_data.programs_2_object[i].to_string(), \
                            file=f
                        )
                print(file=f)

            all_MC_data.append(MC_data)
            
            average.append((MC_data.average_v, program[0].to_string(), program[1].to_string()))
        average.sort(key=lambda tup: tup[0], reverse=True)
        with open(self.folder + 'average_log.txt', 'a') as f:
            print('Average - len = ', len(average), file=f)
            for ave in average:
                print(ave, file=f)
        # Save MC_data to file
        with open(self.folder + 'MC_data', 'wb') as file:
            pickle.dump(all_MC_data, file)
        end_MC = time.time() - start_MC
        with open(self.folder + 'log.txt', 'a') as f:
            print('MC Time elapsed = ', end_MC, file=f)

        #
        # Run Simulated Annealing on the best ones
        #

        #...


        end = time.time()
        print('Running Time: ', end - start, ' seconds.')
        return all_closed_lists

if __name__ == "__main__":
    n_terms = 4
    MC_n_simulations = 4
    n_games = 10
    max_game_rounds = 500
    to_parallel = False
    to_log = False

    start_ISS = time.time()
    ISS = IterativeSketchSynthesizer(n_terms, MC_n_simulations, n_games, max_game_rounds, to_parallel, to_log)
    all_closed_lists = ISS.run()
    end_ISS = time.time() - start_ISS
    print('ISS - Time elapsed = ', end_ISS)