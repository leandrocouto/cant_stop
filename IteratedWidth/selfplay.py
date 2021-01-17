import sys
import random
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import os
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from IteratedWidth.toy_DSL import ToyDSL
from players.random_glenn_player import RandomGlennPlayer
from IteratedWidth.sketch import Sketch
from IteratedWidth.iterated_width import IteratedWidth
from IteratedWidth.breadth_first_search import BFS
from IteratedWidth.simulated_annealing import SimulatedAnnealing
from play_game_template import simplified_play_single_game
from game import Game

class SelfPlay:
    def __init__(self, n_games, n_MC_simulations, max_game_rounds, 
        max_nodes, search_algo, SA, dsl, LS_type, suffix):
        """
          - n_games is the number of games played for evaluation.
          - n_MC_simulations is the number of Monte Carlo simulations done in
            order to evaluate a state.
          - max_game_rounds is the max number of rounds played in a Can't Stop
            game before it is flagged as a draw.
          - max_nodes is the max number of nodes the tree will hold.
          - search_algo is a search algorithm (IW, BFS).
          - SA is a Simulated Annealing algorithm object.
          - dsl is the DSL used to synthesize the programs.
          - LS_type indicates which type of local search is to be used to 
            finish the unfinished programs.
          - suffix is a string for logging purposes to indicate which search
            algorithm is being used.
        """
        self.n_games = n_games
        self.n_MC_simulations = n_MC_simulations
        self.max_game_rounds = max_game_rounds
        self.max_nodes = max_nodes
        self.search_algo = search_algo
        self.SA = SA
        self.dsl = dsl
        self.LS_type = LS_type 
        self.suffix = suffix
        self.filename = os.path.dirname(os.path.realpath(__file__)) + \
                        '/log_file_'+ self.suffix + '.txt'

    def run(self):
        """ Main routine of the selfplay experiment. """

        closed_list = self.search_algo.run()
        # Player initially to be beaten
        player_2 = RandomGlennPlayer()
        player_2_tree = None
        best_player_state = self.local_search(closed_list, player_2)
        #best_player = Sketch(best_player_state.generate_program()).get_object()
        print('best antes')
        print(best_player_state.generate_program())
        SA_state, _ = self.SA.run(best_player_state, player_2)
        print('best depois do SA')
        print(SA_state.generate_program())
        # Save to file
        path = os.path.dirname(os.path.realpath(__file__))
        Sketch(SA_state.generate_program()).save_file_custom(path+'/', self.suffix)
        return SA_state

    def local_search(self, closed_list, player_2):
        """ Apply a local search to all unfinished programs. """

        # Finish the unfinished programs randomly and evaluate them against 
        # player_2
        if self.LS_type == 0:
            scores = []
            for state in closed_list:
                if not state.is_finished():
                    state.finish_tree_randomly()
                player_1 = Sketch(state.generate_program()).get_object()
                try:
                    vic, _, _ = self.play_batch_games(player_1, player_2)
                except:
                    vic = 0
                scores.append(vic)
            print('LS_type = ', self.LS_type)
            print('Best program')
            print(closed_list[np.argmax(scores)].generate_program())
            print('Victories = ', scores[np.argmax(scores)])
            return closed_list[np.argmax(scores)]
        # Finish the unfinished programs randomly, apply SA to each of them
        # and evaluate them against player_2
        elif self.LS_type == 1:
            new_closed_list = []
            scores = []
            for i, state in enumerate(closed_list):
                print('state from CLOSED = ', i)
                print('program')
                print(state.generate_program())
                # Finish randomly all unfinished programs
                if not state.is_finished():
                    state.finish_tree_randomly()
                # Apply SA
                SA_state, _ = self.SA.run(state, player_2)
                new_closed_list.append(SA_state)
                # Evaluate it against player_2
                player_1 = Sketch(SA_state.generate_program()).get_object()
                try:
                    vic, _, _ = self.play_batch_games(player_1, player_2)
                except:
                    vic = 0
                scores.append(vic)
                print('SA victories = ', vic)
                print()
            print('LS_type = ', self.LS_type)
            print('Best program')
            print(new_closed_list[np.argmax(scores)].generate_program())
            print('Victories = ', scores[np.argmax(scores)])
            return new_closed_list[np.argmax(scores)]
            
        # Use Monte Carlo simulations for each of the states while evaluating
        # against player_2
        elif self.LS_type == 2:
            scores = []
            for i, state in enumerate(closed_list):
                print('state from CLOSED = ', i)
                print('program')
                print(state.generate_program())
                local_score = []
                for i in range(self.n_MC_simulations):
                    state_aux = pickle.loads(pickle.dumps(state, -1))
                    if not state_aux.is_finished():
                        state_aux.finish_tree_randomly()
                    player_1 = Sketch(state_aux.generate_program()).get_object()
                    try:
                        vic, _, _ = self.play_batch_games(player_1, player_2)
                    except:
                        vic = 0
                    local_score.append(vic)
                scores.append(sum(local_score)/ len(local_score))
                print('MC simulations score avg = ', sum(local_score)/ len(local_score))
                print()
            # Return the script that won more in the MC simulations
            print('LS_type = ', self.LS_type)
            print('Best program')
            print(closed_list[np.argmax(scores)].generate_program())
            print('Victory avg = ', scores[np.argmax(scores)])
            return closed_list[np.argmax(scores)]



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

    def simplified_play_single_game(self, player1, player2, game, max_game_length):
        is_over = False
        rounds = 0
        # Game loop
        while not is_over:
            rounds += 1
            moves = game.available_moves()
            if game.is_player_busted(moves):
                continue
            else:
                if game.player_turn == 1:
                    chosen_play = player1.get_action(game)
                else:
                    chosen_play = player2.get_action(game)
                # Apply the chosen_play in the game
                print('player - ', game.player_turn,' chosen play = ', chosen_play)
                game.play(chosen_play)
            if rounds > max_game_length:
                who_won = 0
                is_over = True
            else:
                who_won, is_over = game.is_finished()

        return who_won

if __name__ == "__main__":
    suffix = 'IW'
    n_SA_iterations = 30
    n_games = 100
    init_temp = 1
    d = 1
    max_game_rounds = 500
    SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d)
    max_nodes = 50
    n_states = 100
    n_MC_simulations = 100
    k = 4
    # Local Search type. 0 = random, 1 = random + SA, 2 = MC simulations
    LS_type = 1
    #dsl = DSL()
    dsl = ToyDSL()

    # IW
    tree = ParseTree(dsl=dsl, max_nodes=max_nodes, k=k, is_IW=True)
    search_algo = IteratedWidth(tree, n_states, k)

    # BFS
    #tree = ParseTree(dsl=dsl, max_nodes=max_nodes, k=k, is_IW=False)
    #search_algo = BFS(tree, n_states)

    SP = SelfPlay(n_games, n_MC_simulations, max_game_rounds, max_nodes, search_algo, SA, dsl, LS_type, suffix)
    start = time.time()
    SP.run()
    elapsed = time.time() - start
    print('Elapsed = ', elapsed)
    