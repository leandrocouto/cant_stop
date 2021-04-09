import time
import sys
from simulated_annealing import SimulatedAnnealing
from DSL import *
sys.path.insert(0,'..')
from game import Game


if __name__ == "__main__":
    program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
    #incomplete = Argmax(Map(Function(Sum(Map(Function(Minus(Times(HoleNode(), HoleNode()), HoleNode())), NoneNode()))), VarList('actions')))
    #incomplete = Argmax(Map(Function(HoleNode()), VarList('actions')))
    #incomplete = Argmax(Map(Function(Sum(HoleNode())), VarList('actions')))
    #incomplete = Argmax(Map(Function(Sum(Map(Function(HoleNode()), NoneNode()))), VarList('actions')))
    #incomplete = Argmax(Map(Function(Sum(Map(Function(Minus(Times(HoleNode(), HoleNode()), HoleNode())), NoneNode()))), VarList('actions')))
    incomplete = Argmax(Map(HoleNode(), HoleNode()))
    #incomplete = Argmax(HoleNode())

    #initial_program = Rule_of_28_Player_PS(program_yes_no, incomplete)
    n_SA_iterations = 5000
    max_game_rounds = 500
    n_games = 100   
    init_temp = 1
    d = 1
    start_SA = time.time()
    SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d)

    print('incomplete')
    print(incomplete)
    print(incomplete.toString())
    SA.update_parent(incomplete, None)
    SA.curr_id = 0
    SA.id_tree_nodes(incomplete)
    print('arvore')
    incomplete.print_tree()
    print('iniciando sa run')

    best_program, _ = SA.run(incomplete)
    end_SA = time.time() - start_SA
    print('Best program after SA - Time elapsed = ', end_SA)
    print(best_program.toString())
