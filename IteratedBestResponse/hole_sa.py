import time
import sys
from simulated_annealing import SimulatedAnnealing
from DSL import *
sys.path.insert(0,'..')
from game import Game


if __name__ == "__main__":
    program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))

    incomplete = [
                    HoleNode(),
                    Argmax(HoleNode()),
                    Argmax(Map(HoleNode(), HoleNode())),
                    Argmax(Map(Function(HoleNode()), VarList('actions'))),
                    Argmax(Map(Function(Sum(HoleNode())), VarList('actions'))),
                    Argmax(Map(Function(Sum(Map(Function(HoleNode()), NoneNode()))), VarList('actions'))),
                    Argmax(Map(Function(Sum(Map(Function(Minus(Times(HoleNode(), HoleNode()), HoleNode())), NoneNode()))), VarList('actions'))),
                ]
    
    chosen = 3
    n_SA_iterations = 300
    max_game_rounds = 500
    n_games = 100
    init_temp = 1
    d = 1
    algo_name = 'HOLESA_' + str(chosen)
    start_SA = time.time()
    SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name)

    best_program, _ = SA.run(incomplete[chosen])
    end_SA = time.time() - start_SA
    print('Best program after SA - Time elapsed = ', end_SA)
    print(best_program.to_string())
