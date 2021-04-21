import time
import sys
from bottom_up_search import BottomUpSearch
from evaluation import FinishesGame, DefeatsStrategy, DefeatsStrategyNonTriage
from simulated_annealing import SimulatedAnnealing
from DSL import *
sys.path.insert(0,'..')
from game import Game


class IteratedBestResponse:

    def self_play(self):
        start = time.time()
        BUS = BottomUpSearch()
        eval = FinishesGame()
        p, num = BUS.synthesize(
                                bound = 10, 
                                operations = [Sum, Map, Argmax, Function, Plus, Times, Minus], 
                                constant_values = [], 
                                variables_list = [VarList('neutrals'), VarList('actions')], 
                                variables_scalar_from_array = [VarScalarFromArray('progress_value'), VarScalarFromArray('move_value')], 
                                functions_scalars = [NumberAdvancedThisRound(), NumberAdvancedByAction(), IsNewNeutral(), PlayerColumnAdvance(), OpponentColumnAdvance()],
                                eval_function = eval,
                                programs_to_not_eval = set()
                            )
        print('Program that finishes a match in self play')
        print(p.to_string())
        print('programa = ', p)
        
        programs_not_to_eval = set()
        
        for _ in range(10):
            eval = DefeatsStrategy(p)
            
            br, num = BUS.synthesize(
                                bound = 10, 
                                operations = [Sum, Map, Argmax, Function, Plus, Times, Minus], 
                                constant_values = [], 
                                variables_list = [VarList('neutrals'), VarList('actions')], 
                                variables_scalar_from_array = [VarScalarFromArray('progress_value'), VarScalarFromArray('move_value')], 
                                functions_scalars = [NumberAdvancedThisRound(), NumberAdvancedByAction(), IsNewNeutral(), PlayerColumnAdvance(), OpponentColumnAdvance()],
                                eval_function = eval,
                                programs_to_not_eval = set()
                            )
            
            if br is None:
                print('Failed to defeat current strategy... returning current BR.')
                
                end = time.time()
                print('Running Time: ', end - start, ' seconds.')
                
                return p


            
            
            print('Defeats more than 55%: ')
            print(br.to_string())
            print('tree')
            br.print_tree()
            print()
            #print(BUS.get_closed_list())
            print('len closed = ', len(BUS.get_closed_list()))
            programs_not_to_eval = BUS.get_closed_list()
            p = br

            #SA = SimulatedAnnealing(100, 500, 100, 1, 1, False)
            #return SA.run(p)
            #return p
            
        end = time.time()
        print('Running Time: ', end - start, ' seconds.')
        return p

    
'''
IBR = IteratedBestResponse()
p = IBR.self_play()

print('Program encountered: ', p.to_string())
'''
start_IBR = time.time()
IBR = IteratedBestResponse()
p = IBR.self_play()
end_IBR = time.time() - start_IBR
print('IBR - Time elapsed = ', end_IBR)
print(p.to_string())
exit()
n_SA_iterations = 10
max_game_rounds = 500
n_games = 100   
init_temp = 1
d = 1
algo_name = 'IBR'
start_SA = time.time()
SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name)
best_program, _ = SA.run(p)
end_SA = time.time() - start_SA
print('Best program after SA - Time elapsed = ', end_SA)
print(best_program.to_string())