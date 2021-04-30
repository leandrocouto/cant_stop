import time
import sys
import itertools
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
        #return self.all_terms

class IterativeSketchSynthesizer:

    def run(self):
        start = time.time()
        
        partial_DSL = DSLTerms(
                                operations = [HoleNode], 
                                constant_values = [], 
                                variables_list = [], 
                                variables_scalar_from_array = [], 
                                functions_scalars = []
                            )
        
        '''
        partial_DSL = DSLTerms(
                                operations = [Sum, Map, Argmax, Function, Plus, Times, Minus], 
                                constant_values = [], 
                                variables_list = [VarList('neutrals'), VarList('actions')], 
                                variables_scalar_from_array = [VarScalarFromArray('progress_value'), VarScalarFromArray('move_value')], 
                                functions_scalars = [NumberAdvancedThisRound(), NumberAdvancedByAction(), IsNewNeutral(), PlayerColumnAdvance(), OpponentColumnAdvance()]
                            )
        '''
        full_DSL = DSLTerms(
                                operations = [Argmax, Sum, Map, Function, Plus, Times, Minus], 
                                constant_values = [], 
                                variables_list = [VarList('neutrals'), VarList('actions')], 
                                variables_scalar_from_array = [VarScalarFromArray('progress_value'), VarScalarFromArray('move_value')], 
                                functions_scalars = [NumberAdvancedThisRound(), NumberAdvancedByAction(), IsNewNeutral(), PlayerColumnAdvance(), OpponentColumnAdvance()]
                            )
        all_terms = full_DSL.get_all_terms()
        BUS = BottomUpSearch()
        n_terms = len(all_terms)#5
        all_closed_lists = BUS.synthesize(
                                bound = 10, 
                                partial_DSL=partial_DSL,
                                full_DSL=full_DSL,
                                n_terms=n_terms
                            )
        for i in range(len(all_closed_lists)):
            print('i = ', i, ' len = ', len(all_closed_lists[i][1]))
        exit()
        merged_closed_list = all_closed_lists[-1][1]
        #print('merged_closed_list')
        #print(merged_closed_list)
        print('Tamanho antes = ', len(merged_closed_list))
        merged_closed_list = list(itertools.product(merged_closed_list, repeat=2))
        print('Tamanho depois = ', len(merged_closed_list))
        #exit()
        n_simulations = 100
        n_games = 10
        max_game_rounds = 500
        to_parallel = False
        to_log = False

        average = []
        start_MC = time.time()
        for i, program in enumerate(merged_closed_list):
            #print('program = ', program)
            MC = MonteCarloSimulation(program[0], program[1], n_simulations, n_games, max_game_rounds, to_parallel, i, i, to_log)
            v, l, d = MC.run()
            average.append((v, program[0].to_string(), program[1].to_string()))
        average.sort(key=lambda tup: tup[0], reverse=True)
        print('Average - len = ', len(average))
        for ave in average:
            print(ave)
        end_MC = time.time() - start_MC
        print('MC Time elapsed = ', end_MC)

        end = time.time()
        print('Running Time: ', end - start, ' seconds.')
        return all_closed_lists

start_ISS = time.time()
ISS = IterativeSketchSynthesizer()
all_closed_lists = ISS.run()
end_ISS = time.time() - start_ISS
print('ISS - Time elapsed = ', end_ISS)