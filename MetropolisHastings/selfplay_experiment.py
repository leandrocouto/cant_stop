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
from Script import Script
import re

import pickle

class SelfplayExperiment:
    def __init__(self, n_iterations, tree_max_nodes, selfplay_iterations, 
        n_games, victories_needed, max_game_rounds):
        self.n_iterations = n_iterations
        self.tree_max_nodes = tree_max_nodes
        self.selfplay_iterations = selfplay_iterations
        self.n_games = n_games
        self.victories_needed = victories_needed
        self.max_game_rounds = max_game_rounds
        self.file_name = 'seilaporra'

    def generate_player(self, program_string, program_column):
        """ Generate a Player object given the program string. """

        script = Script(
                        program_string, 
                        program_column, 
                        self.n_iterations, 
                        self.tree_max_nodes
                    )
        return self._string_to_object(script._generateTextScript(self.file_name))

    def _string_to_object(self, str_class, *args, **kwargs):
        """ Transform a program written inside str_class to an object. """
        exec(str_class)
        class_name = re.search("class (.*):", str_class).group(1).partition("(")[0]
        return locals()[class_name](*args, **kwargs)

    def run(self):
        
        tree_string_player1 = ParseTree(DSL('S', True), self.tree_max_nodes)
        tree_column_player1 = ParseTree(DSL('S', False), self.tree_max_nodes)
        tree_string_player1.build_tree(tree_string_player1.root)
        tree_column_player1.build_tree(tree_column_player1.root)

        tree_string_player2 = pickle.loads(pickle.dumps(tree_string_player1, -1))
        tree_column_player2 = pickle.loads(pickle.dumps(tree_column_player1, -1))

        tree_string_player2.mutate_tree()
        tree_column_player2.mutate_tree()

        better = 0
        worse = 0
        for i in range(self.selfplay_iterations):
            print('Iteration - ', i)
            
            current_program_string = tree_string_player1.generate_program()
            mutated_program_string = tree_string_player2.generate_program()

            current_program_column = tree_column_player1.generate_program()
            mutated_program_column = tree_column_player2.generate_program()

            script_best_player = self.generate_player(
                                                current_program_string, 
                                                current_program_column
                                                )
            script_mutated_player = self.generate_player(
                                                mutated_program_string,
                                                mutated_program_column
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
                tree_string_player1 = pickle.loads(pickle.dumps(tree_string_player2, -1))
                tree_column_player1 = pickle.loads(pickle.dumps(tree_column_player2, -1))
                print('Mutated was better!')
                better += 1
            else:
                # Mutated program was worse
                # Copy back the first tree (best one) to the second tree
                tree_string_player2 = pickle.loads(pickle.dumps(tree_string_player1, -1))
                tree_column_player2 = pickle.loads(pickle.dumps(tree_column_player1, -1))
                print('Mutated was worse!')
                worse += 1

            # Mutate the second one for the next iteration
            tree_string_player2.mutate_tree()
            tree_column_player2.mutate_tree()

            #t = (victories, losses, draws)
            #print('(win, lose, draw) = ', t)
            #print()
        print('better = ', better)
        print('worse = ', worse)

if __name__ == "__main__":
    n_iterations = 0
    tree_max_nodes = 300
    selfplay_iterations = 100000
    n_games = 2
    victories_needed = 2
    max_game_rounds = 500

    experiment = SelfplayExperiment(
                                    n_iterations, tree_max_nodes, 
                                    selfplay_iterations, n_games, 
                                    victories_needed, max_game_rounds
                                )
    experiment.run()
    





