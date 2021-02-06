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
        max_nodes, search_algo, inner_SA, outer_SA, dsl, LS_type, suffix):
        """
          - n_games is the number of games played for evaluation.
          - n_MC_simulations is the number of Monte Carlo simulations done in
            order to evaluate a state.
          - max_game_rounds is the max number of rounds played in a Can't Stop
            game before it is flagged as a draw.
          - max_nodes is the max number of nodes the tree will hold.
          - search_algo is a search algorithm (IW, BFS).
          - inner_SA is a Simulated Annealing algorithm object used inside
            local search applied to each closed_list state.
          - outer_SA is a Simulated Annealing algorithm object used after
            local_search has decided which state to apply this SA to.
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
        self.inner_SA = inner_SA
        self.outer_SA = outer_SA
        self.dsl = dsl
        self.LS_type = LS_type 
        self.suffix = suffix
        self.folder = os.path.dirname(os.path.realpath(__file__)) + \
                        '/'+self.suffix + '/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.filename = self.folder + '/log_file_'+ self.suffix + '.txt'
        # Log to be used inside SA
        self.inner_SA.folder = self.folder
        self.inner_SA.filename = self.filename
        self.inner_SA.suffix = self.suffix
        self.outer_SA.folder = self.folder
        self.outer_SA.filename = self.filename
        self.outer_SA.suffix = self.suffix

    def run(self):
        """ Main routine of the selfplay experiment. """

        start = time.time()

        closed_list = self.search_algo.run()

        with open(self.folder + 'closed_list', 'wb') as file:
            pickle.dump(closed_list, file)
        # Player initially to be beaten
        player_2 = RandomGlennPlayer()
        # Pass a copy of closed_list to not lose the information of which
        # unfinished program birthed the better program
        copied_closed_list = pickle.loads(pickle.dumps(closed_list, -1))
        # Apply a local search to find the best program in closed_list
        start_LS = time.time()
        best_player_state, best_player_index = self.local_search(copied_closed_list, player_2)
        elapsed_LS = time.time() - start_LS
        best_player = Sketch(best_player_state.generate_program()).get_object()
        # Save to file - Script chosen before local search
        Sketch(closed_list[best_player_index].generate_program()).save_file_custom(self.folder, self.suffix + '_before_LS')
        # Save to file - Script chosen after local search
        Sketch(best_player_state.generate_program()).save_file_custom(self.folder, self.suffix + '_after_LS')
        with open(self.filename, 'a') as f:
            print('Best Script before SA - before transformation', file=f)
            print(closed_list[best_player_index].generate_program(), file=f)
            print('Best Script before SA - after transformation', file=f)
            print(best_player_state.generate_program(), file=f)
            print('Time elapsed for local search =', elapsed_LS, file=f)

        with open(self.filename, 'a') as f:
            print('\n\n\n\n\n\n\n\n Final SA will begin \n\n\n\n\n\n\n\n', file=f)
        # Apply SA to the best player
        start_SA = time.time()
        SA_state, _ = self.outer_SA.run(best_player_state, best_player)
        elapsed_SA = time.time() - start_SA

        with open(self.filename, 'a') as f:
            print('Best Script after SA', file=f)
            print(SA_state.generate_program(), file=f)
            print('Time elapsed for last SA =', elapsed_SA, file=f)
        # Save to file - Final script
        Sketch(SA_state.generate_program()).save_file_custom(self.folder, self.suffix + '_after_SA')

        elapsed = time.time() - start
        with open(self.filename, 'a') as f:
            print('Elapsed time = ', elapsed, file=f)

        return SA_state

    def local_search(self, closed_list, player_2):
        """ Apply a local search to all unfinished programs. """

        # Finish the unfinished programs randomly and evaluate them against 
        # player_2
        if self.LS_type == 0:
            scores = []
            for i, state in enumerate(closed_list):
                start_LS_ite = time.time()
                if not state.is_finished():
                    state.finish_tree_randomly()
                player_1 = Sketch(state.generate_program()).get_object()
                try:
                    vic, _, _ = self.play_batch_games(player_1, player_2)
                except:
                    vic = 0
                scores.append(vic)
                elapsed_LS_ite = time.time() - start_LS_ite
                with open(self.filename, 'a') as f:
                    print('Finished looping through state', i, 'from closed_list in local search. Time =', elapsed_LS_ite, file=f)
            with open(self.filename, 'a') as f:
                print('Score vector = ', scores, file=f)
            best_player_index = np.argmax(scores)
            return closed_list[best_player_index], best_player_index

        # Finish the unfinished programs randomly, apply SA to each of them
        # and evaluate them against player_2
        elif self.LS_type == 1:
            scores = []
            for i, state in enumerate(closed_list):
                start_LS_ite = time.time()
                # Finish randomly all unfinished programs
                if not state.is_finished():
                    state.finish_tree_randomly()
                # Apply SA
                SA_state, _ = self.inner_SA.run(state, player_2)
                closed_list[i] = SA_state
                # Evaluate it against player_2
                player_1 = Sketch(SA_state.generate_program()).get_object()
                try:
                    vic, _, _ = self.play_batch_games(player_1, player_2)
                except:
                    vic = 0
                scores.append(vic)
                elapsed_LS_ite = time.time() - start_LS_ite
                with open(self.filename, 'a') as f:
                    print('Finished looping through state', i, 'from closed_list in local search. Time =', elapsed_LS_ite, file=f)
            with open(self.filename, 'a') as f:
                print('Score vector = ', scores, file=f)
            best_player_index = np.argmax(scores)
            return closed_list[best_player_index], best_player_index
            
        # Use Monte Carlo simulations for each of the states while evaluating
        # against player_2
        elif self.LS_type == 2:
            scores = []
            for i, state in enumerate(closed_list):
                start_LS_ite = time.time()
                local_score = []
                for _ in range(self.n_MC_simulations):
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
                elapsed_LS_ite = time.time() - start_LS_ite
                with open(self.filename, 'a') as f:
                    print('Finished looping through state', i, 'from closed_list in local search. Time =', elapsed_LS_ite, file=f)
            with open(self.filename, 'a') as f:
                print('Score vector (victories avg) = ', scores, file=f)
            # Return the script that won more in the MC simulations
            best_player_index = np.argmax(scores)
            return closed_list[best_player_index], best_player_index



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
                game.play(chosen_play)
            if rounds > max_game_length:
                who_won = 0
                is_over = True
            else:
                who_won, is_over = game.is_finished()

        return who_won

if __name__ == "__main__":
    # Search algorithm -> 0 = IW, 1 = BFS
    search_type = int(sys.argv[1])
    # Local Search algorithm -> 0 = random, 1 = random + SA, 2 = MC simulations
    LS_type = int(sys.argv[2])

    n_SA_iterations_options = [100, 500, 1000]
    n_SA_iterations = n_SA_iterations_options[int(sys.argv[3])]

    # k and n_states are paired together in order to make a fair comparison
    # between algorithms. This is because IW if k is low will have a low number
    # of states returned.
    k_options = [4, 5, 6]
    k = k_options[int(sys.argv[4])]
    n_states_options = [10, 2000, 6000]
    n_states = n_states_options[int(sys.argv[4])]

    n_games = 100
    init_temp = 1
    d = 1
    max_game_rounds = 500
    max_nodes = 200
    n_MC_simulations = 100
    inner_SA = SimulatedAnnealing(n_SA_iterations, max_game_rounds, n_games, init_temp, d, False)
    outer_SA = SimulatedAnnealing(10000, max_game_rounds, n_games, init_temp, d, True)
    

    dsl = ToyDSL()

    '''
    iw = IteratedWidth(ParseTree(dsl=dsl, max_nodes=max_nodes, k=k, is_IW=True), n_states, k)
    closed_iw = iw.run()
    closed_iw = [state.generate_program() for state in closed_iw]
    bfs = BFS(ParseTree(dsl=dsl, max_nodes=max_nodes, k=k, is_IW=False), n_states)
    closed_bfs = bfs.run()
    closed_bfs = [state.generate_program() for state in closed_bfs]
    equal = 0
    for program in closed_iw:
        if program in closed_bfs:
            equal += 1
    print('Programas em comum = ', equal)
    '''
    
    if search_type == 0:
        # IW
        tree = ParseTree(dsl=dsl, max_nodes=max_nodes, k=k, is_IW=True)
        search_algo = IteratedWidth(tree, n_states, k)
        suffix = 'IW' + '_LS' + str(LS_type) + '_SA' + str(n_SA_iterations) + '_ST' + str(n_states) + '_k' + str(k) + '_GA' + str(n_games)
    elif search_type == 1:
        # BFS
        tree = ParseTree(dsl=dsl, max_nodes=max_nodes, k=k, is_IW=False)
        search_algo = BFS(tree, n_states)
        suffix = 'BFS' + '_LS' + str(LS_type) + '_SA' + str(n_SA_iterations) + '_ST' + str(n_states) + '_k' + str(k) + '_GA' + str(n_games)

    SP = SelfPlay(n_games, n_MC_simulations, max_game_rounds, max_nodes, search_algo, inner_SA, outer_SA, dsl, LS_type, suffix)
    start = time.time()
    SP.run()
    elapsed = time.time() - start
    print('Elapsed = ', elapsed)
    
    
    