import sys
import random
import time
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from IteratedWidth.toy_DSL import ToyDSL
from IteratedWidth.sketch import Sketch
from IteratedWidth.iterated_width import IteratedWidth
from play_game_template import simplified_play_single_game
from game import Game

class SelfPlay:
    def __init__(self, n_selfplay_iterations, n_games, max_game_rounds, IW, dsl):
        self.n_selfplay_iterations = n_selfplay_iterations
        self.n_games = n_games
        self.max_game_rounds = max_game_rounds
        self.IW = IW
        self.dsl = dsl

    def run(self):
        open_list = self.IW.run()
        print('size antes = ', len(open_list))
        open_list = self.remove_unfinished_programs(open_list)
        print('size dps = ', len(open_list))
        _, player_2 = self.generate_random_program()
        for i in range(self.n_selfplay_iterations):
            scores = []
            for j in range(len(open_list)):
                program = Sketch(open_list[j].generate_program()).get_object()
                try:
                    victories = self.play_batch_games(program, player_2)
                    scores.append(victories)
                except Exception as e:
                    print('excecao - ', j, ' - ', str(e))
                    continue
            best_player_index = np.argmax(scores)
            if scores[best_player_index] > self.n_games // 2:
                player_2 = open_list[best_player_index]
        return player_2

    def play_batch_games(self, player_1, player_2):

        victories = 0
        losses = 0
        draws = 0

        for i in range(self.n_games):
            game = Game(2, 4, 6, [2,12], 2, 2)
            if i%2 == 0:
                    who_won = simplified_play_single_game(
                                                        player_1, 
                                                        player_2, 
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
                                                    player_2, 
                                                    player_1, 
                                                    game, 
                                                    self.max_game_rounds
                                                )
                if who_won == 2:
                    victories += 1
                elif who_won == 1:
                    losses += 1
                else:
                    draws += 1

        return victories

    def generate_random_program(self):
        """ Create an initial random synthesized program. """

        tree = ParseTree(self.dsl, tree_max_nodes, k, False)
        tree.build_random_tree(tree.root)
        program_string = tree.generate_program()
        sketch = Sketch(program_string)
        program_object = sketch.get_object()

        return tree, program_object

    def remove_unfinished_programs(self, open_list):
        """ 
        Remove all programs that still have 'unfinished nodes' in it. Note 
        that this does not guarantee that the program is actually executable.
        """
        '''
        print('open_list')
        print(open_list)
        for state in open_list:
            print('estado programa')
            print(state.generate_program())
            print()
        '''
        finished_programs = []  
        for state in open_list:
            if state.is_finished():
                finished_programs.append(state)
        return finished_programs

if __name__ == "__main__":
    tree_max_nodes = 50
    n_expansions = 500
    k = 15
    n_selfplay_iterations = 100
    n_games = 1000
    max_game_rounds = 500
    #dsl = DSL()
    dsl = ToyDSL()
    tree = ParseTree(dsl, tree_max_nodes, k, True)
    IW = IteratedWidth(tree, n_expansions, k)
    SF = SelfPlay(n_selfplay_iterations, n_games, max_game_rounds, IW, dsl)
    SF.run()
    