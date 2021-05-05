import time
import sys
from simulated_annealing import SimulatedAnnealing
from DSL import *
sys.path.insert(0,'..')
from game import Game


if __name__ == "__main__":

    incomplete = [
                    HoleNode(),
                    Argmax(HoleNode()),
                    Argmax(Map(HoleNode(), HoleNode())),
                    Argmax(Map(Function(HoleNode()), VarList('actions'))),
                    Argmax(Map(Function(Sum(HoleNode())), VarList('actions'))),
                    Argmax(Map(Function(Sum(Map(Function(HoleNode()), NoneNode()))), VarList('actions'))),
                    Argmax(Map(Function(Sum(Map(Function(Minus(Times(HoleNode(), HoleNode()), HoleNode())), NoneNode()))), VarList('actions'))),
                ]
    run = int(sys.argv[1])
    chosen = 0
    n_SA_iterations = 100000
    max_game_rounds = 500
    n_games = 1000
    init_temp = 1
    d = 1
    initial_time = 0.0
    algo_name = 'run_' + str(run) + '_SA'
    start_SA = time.time()
    SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, algo_name, initial_time)
    is_complete = False
    _, _, _, _, _, _ = SA.run(incomplete[chosen], incomplete[chosen], is_complete)
    end_SA = time.time() - start_SA
    print('Time elapsed = ', end_SA)
