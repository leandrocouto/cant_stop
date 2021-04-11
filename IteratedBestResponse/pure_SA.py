import time
import sys
from bottom_up_search import BottomUpSearch
from evaluation import FinishesGame, DefeatsStrategy, DefeatsStrategyNonTriage
from simulated_annealing import SimulatedAnnealing
from DSL import *
sys.path.insert(0,'..')
from game import Game


if __name__ == "__main__":
    start_BUS = time.time()
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
    end_BUS = time.time() - start_BUS
    print('BUS program that finishes a match - Time elapsed = ', end_BUS)
    print(p.to_string())
    n_SA_iterations = 2000
    max_game_rounds = 500
    n_games = 100   
    init_temp = 1
    d = 1
    algo_name = 'PURESA'
    start_SA = time.time()
    SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name)
    best_program, _ = SA.run(p)
    end_SA = time.time() - start_SA
    print('Best program after SA - Time elapsed = ', end_SA)
    print(best_program.to_string())
