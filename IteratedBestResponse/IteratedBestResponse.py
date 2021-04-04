import time
import sys
from BottomUpSearch import BottomUpSearch
from Evaluation import FinishesGame, DefeatsStrategy, DefeatsStrategyNonTriage
from simulated_annealing import SimulatedAnnealing
from DSL import *
sys.path.insert(0,'..')
from game import Game


class IteratedBestResponse:

    def self_play(self):
        start = time.time()
        bus = BottomUpSearch()
        eval = FinishesGame()
        p, num = bus.synthesize(
                                bound = 10, 
                                operations = [Sum, Map, Argmax, Function, Plus, Times, Minus], 
                                constant_values = [], 
                                variables_list = ['neutrals', 'actions'], 
                                variables_scalar_from_array = ['progress_value', 'move_value'], 
                                functions_scalars = [NumberAdvancedThisRound, NumberAdvancedByAction, IsNewNeutral],
                                eval_function = eval,
                                programs_to_not_eval = set()
                            )
        print('Program that finishes a match in self play')
        print(p.toString())
        print('programa = ', p)
        
        programs_not_to_eval = set()
        
        for _ in range(10):
            eval = DefeatsStrategy(p)
#             eval = DefeatsStrategyNonTriage(p)
            
            br, num = bus.synthesize(
                        bound = 10, 
                        operations = [Sum, Map, Argmax, Function, Plus, Times, Minus], 
                        constant_values = [], 
                        variables_list = ['neutrals', 'actions'], 
                        variables_scalar_from_array = ['progress_value', 'move_value'], 
                        functions_scalars = [NumberAdvancedThisRound, NumberAdvancedByAction, IsNewNeutral],
                        eval_function = eval,
                        programs_to_not_eval = programs_not_to_eval
                    )
            
            if br is None:
                print('Failed to defeat current strategy... returning current BR.')
                
                end = time.time()
                print('Running Time: ', end - start, ' seconds.')
                
                return p


            
            
            print('Defeats more than 55%: ')
            print(br.toString())
            print('tree')
            br.print_tree(br, '  ')
            print()
            #print(bus.get_closed_list())
            print('len closed = ', len(bus.get_closed_list()))
            programs_not_to_eval = bus.get_closed_list()
            p = br

            #SA = SimulatedAnnealing(100, 500, 100, 1, 1, False)
            #return SA.run(p)
            #return p
            
        end = time.time()
        print('Running Time: ', end - start, ' seconds.')
        return p

    
'''
ibr = IteratedBestResponse()
p = ibr.self_play()

print('Program encountered: ', p.toString())
'''
start_IBR = time.time()
ibr = IteratedBestResponse()
p = ibr.self_play()
end_IBR = time.time() - start_IBR
print('IBR - Time elapsed = ', end_IBR)
print(p.toString())
n_SA_iterations = 2000
max_game_rounds = 500
n_games = 100   
init_temp = 1
d = 1
start_SA = time.time()
#SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d)
#best_program, _ = SA.run(p)
end_SA = time.time() - start_SA
print('Best program after SA - Time elapsed = ', end_SA)
#print(best_program.toString())