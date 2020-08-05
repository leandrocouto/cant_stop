import sys
sys.path.insert(0,'..')
from play_game_template import simplified_play_single_game
from players.vanilla_uct_player import Vanilla_UCT
from players.rule_of_28_player import Rule_of_28_Player
from MetropolisHastings.metropolis_hastings import MetropolisHastings
from MetropolisHastings.simulated_annealing import SimulatedAnnealing

from MetropolisHastings.parse_tree import ParseTree, Node
from MetropolisHastings.DSL import DSL
from game import Game

import pickle

class SelfplayExperiment:
    def __init__(self, k, n_iterations, tree_max_nodes, selfplay_iterations, 
        n_games, victories_needed, max_game_rounds):
        self.k = k
        self.n_iterations = n_iterations
        self.tree_max_nodes = tree_max_nodes
        self.selfplay_iterations = selfplay_iterations
        self.n_games = n_games
        self.victories_needed = victories_needed
        self.max_game_rounds = max_game_rounds

    def run(self):
        
        opt_algo = MetropolisHastings(
                                    0.5,
                                    100, 
                                    self.k,
                                    0,
                                    self.tree_max_nodes,
                                    'fulldata_sorted'
                                )

        _, _, player_1_tree = opt_algo.run()
        #player_1_tree = ParseTree(DSL('S'), self.tree_max_nodes)
        #player_1_tree.build_tree(player_1_tree.root)
        player_2_tree = pickle.loads(pickle.dumps(player_1_tree, -1))

        player_2_tree.mutate_tree()

        for i in range(self.selfplay_iterations):
            print('Iteration - ', i)
            current_program = player_1_tree.generate_program()
            mutated_program = player_2_tree.generate_program()
            script_best_player = player_1_tree.generate_player(
                                                        current_program, 
                                                        self.k, 
                                                        self.n_iterations, 
                                                        self.tree_max_nodes
                                                        )
            script_mutated_player = player_2_tree.generate_player(
                                                        current_program, 
                                                        self.k, 
                                                        self.n_iterations, 
                                                        self.tree_max_nodes
                                                        )
            victories = 0
            losses = 0
            draws = 0
            for j in range(self.n_games):
                game = game = Game(2, 4, 6, [2,12], 2, 2)
                if j%2 == 0:
                        who_won = simplified_play_single_game(
                                                            script_best_player, 
                                                            script_mutated_player, 
                                                            game, 
                                                            self.max_game_rounds
                                                        )
                        if who_won == 1:
                            victories += 1
                        elif who_won == 2:
                            losses += 1
                        else:
                            draws += 1
                else:
                    who_won = simplified_play_single_game(
                                                        script_mutated_player, 
                                                        script_best_player, 
                                                        game, 
                                                        self.max_game_rounds
                                                    )
                    if who_won == 2:
                        victories += 1
                    elif who_won == 1:
                        losses += 1
                    else:
                        draws += 1

            if victories >= self.victories_needed:
                # Mutated program was better.
                # Copy the mutated tree as the best one
                player_1_tree = pickle.loads(pickle.dumps(player_2_tree, -1))
                print('mutated was better!')
            else:
                # Mutated program was worse
                # Copy back the first tree (best one) to the second tree
                player_2_tree = pickle.loads(pickle.dumps(player_1_tree, -1))
                print('mutated was worse!')

            # Mutate the second one for the next iteration
            player_2_tree.mutate_tree()
            t = (victories, losses, draws)
            print('(win, lose, draw) = ', t)
            print()

if __name__ == "__main__":
    k = 0
    n_iterations = 0
    tree_max_nodes = 300
    selfplay_iterations = 100
    n_games = 100
    victories_needed = 55
    max_game_rounds = 500

    experiment = SelfplayExperiment(
                                    k, n_iterations, tree_max_nodes, 
                                    selfplay_iterations, n_games, 
                                    victories_needed, max_game_rounds
                                )
    experiment.run()
    





