import time
import sys
import itertools
import os
import pickle
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

    def get_all_terms(self):
        self.all_terms = [op.className() for op in self.operations] + \
                        [op.className() for op in self.constant_values] + \
                        [op.className() for op in self.variables_list] + \
                        [op.className() for op in self.variables_scalar_from_array] + \
                        [op.className() for op in self.functions_scalars]
        # No duplicates, maintain order
        return list(dict.fromkeys(self.all_terms))

class IterativeSketchSynthesizer:
    def __init__(self, n_terms, MC_n_simulations, n_games, max_game_rounds, to_parallel, to_log, use_average):
        self.n_terms = n_terms
        self.MC_n_simulations = MC_n_simulations
        self.n_games = n_games
        self.max_game_rounds = max_game_rounds
        self.to_parallel = to_parallel
        self.to_log = to_log
        self.use_average = use_average
        if self.use_average:
            self.folder = 'ISS_' + str(self.n_terms) + 'terms_' + str(self.MC_n_simulations) + \
                            'sim_' + str(self.n_games) + 'games_use_average/'
        else:
            self.folder = 'ISS_' + str(self.n_terms) + 'terms_' + str(self.MC_n_simulations) + \
                            'sim_' + str(self.n_games) + 'games_use_best/'
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
        victory = []
        #all_MC_data = []
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
                print('Best programs simulated by MC', file=f)

                # Stores the best (by number of victories) of each MC sim.
                best_vic = []
                for i in range(len(MC_data.simulation_victories[:3])):
                    #best_vic.append((MC_data.simulation_victories[i], program[0].to_string(), program[1].to_string(), MC_data.programs_1_object[i].to_string(), MC_data.programs_2_object[i].to_string()))
                    best_vic.append(
                                    (
                                        program[0].to_string(), 
                                        program[1].to_string(), 
                                        MC_data.simulation_victories[i], 
                                        MC_data.simulation_losses[i], 
                                        MC_data.simulation_draws[i], 
                                        MC_data.programs_1_object[i], 
                                        MC_data.programs_2_object[i]
                                    )
                                )
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
        # Save best sketches (in terms of average) to file 
        average.sort(key=lambda tup: tup[2], reverse=True)
        with open(self.folder + 'average_log.txt', 'a') as f:
            print('Average - len = ', len(average), file=f)
            for ave in average:
                print('Sketch 1 and 2 = ', ave[0].to_string(), ave[1].to_string(), file=f)
                print('Average V/L/D = ', ave[2], ave[3], ave[4], file=f)
                print('Stdev V/L/D = ', ave[5], ave[6], ave[7], file=f)
                print(file=f)
        # Save best programs (in terms of victory) to file 
        victory.sort(key=lambda tup: tup[0][2], reverse=True)
        with open(self.folder + 'victory_log.txt', 'a') as f:
            print('Victory - len = ', len(victory), file=f)
            for vic_best in victory:
                print('Sketch = ', vic_best[0][0], vic_best[0][1], file=f)
                for vic in vic_best:
                    print('V/L/D = ', vic[2], vic[3], vic[4], vic[5].to_string(), vic[6].to_string(), file=f)
                print(file=f)
        # Save both victory and average to file
        with open(self.folder + 'average', 'wb') as file:
            pickle.dump(average, file)
        with open(self.folder + 'victory', 'wb') as file:
            pickle.dump(victory, file)
        end_MC = time.time() - start_MC
        with open(self.folder + 'log.txt', 'a') as f:
            print('MC Time elapsed = ', end_MC, file=f)

        #
        # Run Simulated Annealing on the best ones
        #
        best_average_1 = average[0][0]
        best_average_2 = average[0][1]

        best_victory_1 = victory[0][0][5]
        best_victory_2 = victory[0][0][6]

        print('best_average_1')
        print(best_average_1.to_string())
        best_average_1.print_tree()
        print('best_average_2')
        print(best_average_2.to_string())
        best_average_2.print_tree()
        print('best_victory_1')
        print(best_victory_1.to_string())
        best_victory_1.print_tree()
        print('best_victory_2')
        print(best_victory_2.to_string())
        best_victory_2.print_tree()

        n_SA_iterations = 20000
        max_game_rounds = 500
        n_games = 1000
        init_temp = 1
        d = 1
        algo_name = self.folder
        initial_time = time.time() - start
        start_SA = time.time()
        SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name, initial_time)
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
    n_terms = 4
    MC_n_simulations = 100
    n_games = 10
    max_game_rounds = 500
    to_parallel = False
    to_log = False
    use_average = False
    start_ISS = time.time()
    ISS = IterativeSketchSynthesizer(n_terms, MC_n_simulations, n_games, max_game_rounds, to_parallel, to_log, use_average)
    all_closed_lists = ISS.run()
    end_ISS = time.time() - start_ISS
    print('ISS - Time elapsed = ', end_ISS)