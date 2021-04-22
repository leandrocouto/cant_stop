import time
import sys
from partial_bottom_up_search import PartialBottomUpSearch
from evaluation import FinishesGame, DefeatsStrategy, DefeatsStrategyNonTriage
from simulated_annealing import SimulatedAnnealing
from partial_DSL import *
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
        PBUS = PartialBottomUpSearch()
        all_closed_lists = PBUS.synthesize(
                                bound = 10, 
                                partial_DSL=partial_DSL,
                                full_DSL=full_DSL
                            )
            
        end = time.time()
        print('Running Time: ', end - start, ' seconds.')
        return all_closed_lists

    
'''
IBR = IteratedBestResponse()
p = IBR.self_play()

print('Program encountered: ', p.to_string())
'''
start_ISS = time.time()
ISS = IterativeSketchSynthesizer()
all_closed_lists = ISS.run()
end_ISS = time.time() - start_ISS
print('ISS - Time elapsed = ', end_ISS)
exit()
n_SA_iterations = 10
max_game_rounds = 500
n_games = 100   
init_temp = 1
d = 1
algo_name = 'ISS'
start_SA = time.time()
SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name)
best_program, _ = SA.run(p)
end_SA = time.time() - start_SA
print('Best program after SA - Time elapsed = ', end_SA)
print(best_program.to_string())