import sys
import random
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from IteratedWidth.toy_DSL import ToyDSL
from IteratedWidth.sketch import Sketch
from IteratedWidth.iterated_width import IteratedWidth
from IteratedWidth.simulated_annealing import SimulatedAnnealing
from play_game_template import simplified_play_single_game
from game import Game

class SelfPlay:
    def __init__(self, n_selfplay_iterations, n_games, max_game_rounds, IW, SA, 
        dsl, LS_type):
        """
          - n_games is the number of games played for evaluation.
          - max_game_rounds is the max number of rounds played in a Can't Stop
            game before it is flagged as a draw.
          - IW is an Iterated Width algorithm object.
          - SA is a Simulated Annealing algorithm object.
          - dsl is the DSL used to synthesize the programs.
          - LS_type indicates which type of local search is to be used to 
            finish the unfinished programs.
        """
        self.n_selfplay_iterations = n_selfplay_iterations
        self.n_games = n_games
        self.max_game_rounds = max_game_rounds
        self.IW = IW
        self.SA = SA
        self.dsl = dsl
        self.LS_type = LS_type 

    def run(self):
        """ Main routine of the selfplay experiment. """

        open_list, _ = self.IW.run()
        #open_list = self.remove_unfinished_programs(open_list)
        player_2, player_2_tree = self.generate_random_program()
        print('antes local search')
        open_list = self.local_search(open_list, player_2_tree)
        print('dps local search')
        exit()
        for i in range(self.n_selfplay_iterations):
            print('iteracao selfplay - ', i)
            victories = []
            for j in range(len(open_list)):
                program = Sketch(open_list[j].generate_program()).get_object()
                try:
                    vic, loss, draw = self.play_batch_games(program, player_2)
                    victories.append(vic)
                    #print('V/L/D = ', vic, loss, draw)
                except Exception as e:
                    print('excecao - ', j, ' - ', str(e))
                    victories.append(0)
                    continue
            print('victories = ', victories)
            best_player_index = np.argmax(victories)
            if victories[best_player_index] > self.n_games // 2:
                player_2 = open_list[best_player_index]
        return player_2

    def generate_random_program(self):
        """ Create an initial random synthesized program. """

        tree = ParseTree(self.dsl, tree_max_nodes, k, False)
        tree.build_random_tree(tree.root)
        program_string = tree.generate_program()
        sketch = Sketch(program_string)
        program_object = sketch.get_object()

        return program_object, tree 

    def local_search(self, open_list, player_2_state):
        """ Apply a local search to all unfinished programs. """

        # Finish the unfinished programs randomly
        if self.LS_type == 0:
            for state in open_list:
                if not state.is_finished():
                    state.finish_tree_randomly()
            return open_list
        # Finish the unfinished program randomly and then apply SA
        elif self.LS_type == 1:
            if_counter = 0
            else_counter = 0
            new_open_list = []
            for i in range(len(open_list)):
                print('iniciando SA iteracao - ', i)
                if not open_list[i].is_finished():
                    open_list[i].finish_tree_randomly()

                    SA_state, _ = self.SA.run(open_list[i], player_2_state)
                    new_open_list.append(SA_state)
                    if_counter += 1
                else:
                    SA_state, _ = self.SA.run(open_list[i], player_2_state)
                    new_open_list.append(SA_state)
                    else_counter += 1
            print('if = ', if_counter)
            print('else = ', else_counter)
            return new_open_list
        else:
            pass

    def remove_unfinished_programs(self, open_list):
        """ 
        Remove all programs that still have 'unfinished nodes' in it. Note 
        that this does not guarantee that the program is actually executable.
        """

        finished_programs = []  
        for state in open_list:
            if state.is_finished():
                finished_programs.append(state)
        return finished_programs

    def play_batch_games(self, player_1, player_2):
        """ 
        Play self.n_games Can't Stop games between player_1 and player_2. The
        players swap who is the first player per iteration because Can't Stop
        is biased towards the player who plays the first move.
        """
        
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

        return victories, losses, draws

if __name__ == "__main__":
    n_SA_iterations = 50
    n_games = 10
    init_temp = 1
    d = 1
    SA = SimulatedAnnealing(n_SA_iterations, n_games, init_temp, d)
    tree_max_nodes = 50
    n_expansions = 10
    k = 10
    n_selfplay_iterations = 10
    max_game_rounds = 500
    # Local Search type. 0 = random, 1 = random + SA
    LS_type = 1
    #dsl = DSL()
    dsl = ToyDSL()
    tree = ParseTree(dsl, tree_max_nodes, k, True)
    IW = IteratedWidth(tree, n_expansions, k)
    SF = SelfPlay(n_selfplay_iterations, n_games, max_game_rounds, IW, SA, dsl, LS_type)
    SF.run()
    