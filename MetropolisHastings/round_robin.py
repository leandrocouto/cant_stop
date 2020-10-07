import sys
sys.path.insert(0,'..')
from players.player import Player
from players.random_player import RandomPlayer
from players.wrong_glenn_player import Wrong_Glenn_Player
from players.new_glenn_player import New_Glenn_Player
from players.correct_glenn_player import Correct_Glenn_Player
from players.glenn_player import Glenn_Player
from players.rule_of_28 import Rule_of_28
from players.vanilla_uct_player import Vanilla_UCT
from MetropolisHastings.DSL import DSL
import random
import pickle
import time
from game import Game
from Script import Script
from play_game_template import play_single_game
from play_game_template import simplified_play_single_game
from itertools import combinations 

class RandomWalkSelfplay(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = 34 + DSL.number_cells_advanced_this_round(state) 
            if score_yes_no < 39 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 12 , 1 , 12 , 12 , 6 , 13 , 18 , 3 , 9 , 18 , 5 ] 
            for i in range(len(actions)): 
                score_columns[i] = 7 + DSL.advance_in_action_col(actions[i]) + DSL.get_weight_for_action_columns(actions[i], weights) - abs( DSL.does_action_place_new_neutral(actions[i], state) * 35 ) 
            return actions[np.argmax(score_columns)] 

class FictitiousPlay(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = DSL.get_player_score(state) - abs( 33 - 32 ) + DSL.number_cells_advanced_this_round(state) * DSL.get_player_score(state) + DSL.get_player_score(state) - 34 * abs( 37 - 6 - 38 ) + DSL.number_cells_advanced_this_round(state) * 13 
            if score_yes_no < 16 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 14 , 3 , 3 , 13 , 19 , 14 , 12 , 3 , 19 , 7 , 13 ] 
            for i in range(len(actions)): 
                score_columns[i] = abs( 1 - DSL.get_weight_for_action_columns(actions[i], weights) ) 
            return actions[np.argmax(score_columns)] 

class RandomWalkFictitiousPlay(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = abs( 32 - abs( abs( 39 - DSL.number_cells_advanced_this_round(state) * 23 - DSL.get_player_score(state) + 37 - DSL.get_opponent_score(state) ) - 37 + DSL.get_player_score(state) ) ) 
            if score_yes_no < 39 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 10 , 5 , 14 , 3 , 17 , 14 , 9 , 7 , 17 , 7 , 17 ] 
            for i in range(len(actions)): 
                score_columns[i] = abs( DSL.get_weight_for_action_columns(actions[i], weights) - abs( DSL.advance_in_action_col(actions[i]) * 7 ) + 19 - DSL.advance_in_action_col(actions[i]) ) + 33 + 12 
            return actions[np.argmax(score_columns)] 

class SimulatedAnnealingSelfplay(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = abs( 36 - DSL.number_cells_advanced_this_round(state) - abs( abs( 7 - DSL.number_cells_advanced_this_round(state) ) - 3 - 16 ) - 5 * DSL.number_cells_advanced_this_round(state) + 14 - DSL.get_player_score(state) ) 
            if score_yes_no < 28 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 2 , 8 , 11 , 1 , 17 , 19 , 11 , 11 , 11 , 5 , 9 ] 
            for i in range(len(actions)): 
                score_columns[i] = DSL.get_weight_for_action_columns(actions[i], weights) 
            return actions[np.argmax(score_columns)] 

class BoostedFictitiousPlay(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = abs( abs( abs( DSL.get_player_score(state) + DSL.get_player_score(state) ) + DSL.get_player_score(state) ) + abs( 22 * DSL.number_cells_advanced_this_round(state) + 31 * 9 ) ) - 33 - 11 * 39 - DSL.number_cells_advanced_this_round(state) 
            if score_yes_no < 13 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 9 , 1 , 8 , 2 , 17 , 13 , 18 , 2 , 11 , 5 , 11 ] 
            for i in range(len(actions)): 
                score_columns[i] = DSL.get_weight_for_action_columns(actions[i], weights) - DSL.advance_in_action_col(actions[i]) * DSL.does_action_place_new_neutral(actions[i], state) * 9 
            return actions[np.argmax(score_columns)] 

class BoostedSimulatedAnnealingSelfplay(Player):
    def get_action(self, state):
        import numpy as np
        actions = state.available_moves()
        score_yes_no = 0
        score_columns = np.zeros(len(actions))
        if actions[0] in ['y','n'] : 
            score_yes_no = abs( 28 - abs( abs( 33 - abs( DSL.get_player_score(state) * DSL.number_cells_advanced_this_round(state) ) ) - 29 ) ) 
            if score_yes_no < 35 : 
                return 'y' 
            else: 
                return 'n' 
        else: 
            score_columns = np.zeros(len(actions)) 
            weights = [ 12 , 1 , 2 , 6 , 6 , 13 , 8 , 2 , 15 , 2 , 1 ] 
            for i in range(len(actions)): 
                score_columns[i] = abs( DSL.advance_in_action_col(actions[i]) + abs( abs( DSL.get_weight_for_action_columns(actions[i], weights) * 9 ) - 1 ) - DSL.does_action_place_new_neutral(actions[i], state) ) 
            return actions[np.argmax(score_columns)] 

RandomWalkSelfplay = RandomWalkSelfplay()
SimulatedAnnealingSelfplay = SimulatedAnnealingSelfplay()
FictitiousPlay = FictitiousPlay()
RandomWalkFictitiousPlay = RandomWalkFictitiousPlay()
BoostedFictitiousPlay = BoostedFictitiousPlay()
BoostedSimulatedAnnealingSelfplay = BoostedSimulatedAnnealingSelfplay()
Correct_Glenn_Player = Correct_Glenn_Player()
Wrong_Glenn_Player = Wrong_Glenn_Player()
Glenn_Player = Glenn_Player()
New_Glenn_Player = New_Glenn_Player()
Rule_of_28 = Rule_of_28()

players = [
    (Glenn_Player, "Glenn Player", "GLENN"),
    (Rule_of_28, "Rule of 28", "RULE28"),
    #(New_Glenn_Player, "New Glenn Player", "NEWGLENN")
    #(Wrong_Glenn_Player, "Wrong_Glenn_Player", "WRONG"),
    #(Correct_Glenn_Player, "Correct_Glenn_Player", "CORRECT"),
    #(RandomWalkSelfplay, "Random Walk Selfplay", "RWSP"),
    #(SimulatedAnnealingSelfplay, "Simulated Annealing Selfplay", "SASP"),
    #(FictitiousPlay, "Fictitious Play", "SAFP"),
    #(RandomWalkFictitiousPlay, "Random Walk Fictitious Play", "RWFP"),
    #(BoostedFictitiousPlay, "Boosted Fictitious Play", "BSAFP"),
    #(BoostedSimulatedAnnealingSelfplay, "Boosted Simulated Annealing Selfplay", "BSASP")

]

n_games = 10000

combs = combinations([i for i in range(len(players))], 2)
combs = list(combs)


# Alternating who is first player
for comb in combs: 
    player1 = players[comb[0]][0]
    player2 = players[comb[1]][0]
    if players[comb[0]][2] != "GLENN" and players[comb[1]][2] != "GLENN":
        continue
    filename = "round_robin_" + players[comb[0]][2] + "_vs_" + players[comb[1]][2] + '.txt'

    with open(filename, 'a') as f:
        print(players[comb[0]][1] + " vs " + players[comb[1]][1], file=f)

    v = 0
    l = 0
    d = 0
    for j in range(n_games):
        start = time.time()
        game = game = Game(2, 4, 6, [2,12], 2, 2)
        if j%2 == 0:
            who_won = simplified_play_single_game(player1, player2, game, 5000000)
            if who_won == 1:
                v += 1
                with open(filename, 'a') as f:
                    print('Iteration ', j, players[comb[0]][1], ' won', file=f)
            elif who_won == 2:
                l += 1
                with open(filename, 'a') as f:
                    print('Iteration ', j, players[comb[1]][1], ' won', file=f)
            else:
                d += 1
                with open(filename, 'a') as f:
                    print('Iteration ', j, ' - Draw', file=f)
        else:
            who_won = simplified_play_single_game(player2, player1, game, 5000000)
            if who_won == 2:
                v += 1
                with open(filename, 'a') as f:
                    print('Iteration ', j, players[comb[1]][1], ' won', file=f)
            elif who_won == 1:
                l += 1
                with open(filename, 'a') as f:
                    print('Iteration ', j, players[comb[0]][1], ' won', file=f)
            else:
                d += 1
                with open(filename, 'a') as f:
                    print('Iteration ', j, ' - Draw', file=f)

        end = time.time() - start
        with open(filename, 'a') as f:
            print('V/L/D = ', v, l, d, ' - Iteration time = ', end, file=f)

    with open(filename, 'a') as f:
        print(players[comb[0]][1], 'won', v, 'matches', file=f)
        print(players[comb[1]][1], 'won', l, 'matches', file=f)
        print('The match was drawn', d, 'times', file=f)
    print(players[comb[0]][1], 'won', v, 'matches')
    print(players[comb[1]][1], 'won', l, 'matches')
    print('The match was drawn', d, 'times')
    print()

'''
# Not  alternating who is first player
for comb in combs: 
    player1 = players[comb[0]][0]
    player2 = players[comb[1]][0]
    if players[comb[0]][2] != "GLENN" and players[comb[1]][2] != "GLENN":
        continue
    filename = "round_robin_" + players[comb[0]][2] + "_vs_" + players[comb[1]][2] + '.txt'

    with open(filename, 'a') as f:
        print(players[comb[0]][1] + " vs " + players[comb[1]][1], file=f)

    v = 0
    l = 0
    d = 0
    for j in range(n_games):
        start = time.time()
        game = game = Game(2, 4, 6, [2,12], 2, 2)
        who_won = simplified_play_single_game(player1, player2, game, 5000000)
        if who_won == 1:
            v += 1
            with open(filename, 'a') as f:
                print('Iteration ', j, players[comb[0]][1], ' won', file=f)
        elif who_won == 2:
            l += 1
            with open(filename, 'a') as f:
                print('Iteration ', j, players[comb[1]][1], ' won', file=f)
        else:
            d += 1
            with open(filename, 'a') as f:
                print('Iteration ', j, ' - Draw', file=f)
        end = time.time() - start
        with open(filename, 'a') as f:
            print('V/L/D = ', v, l, d, ' - Iteration time = ', end, file=f)

    with open(filename, 'a') as f:
        print(players[comb[0]][1], 'won', v, 'matches', file=f)
        print(players[comb[1]][1], 'won', l, 'matches', file=f)
        print('The match was drawn', d, 'times', file=f)
    print(players[comb[0]][1], 'won', v, 'matches')
    print(players[comb[1]][1], 'won', l, 'matches')
    print('The match was drawn', d, 'times')
    print()
'''

