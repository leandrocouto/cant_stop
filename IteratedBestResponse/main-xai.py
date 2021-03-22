import random
import numpy as np
import random
import collections
import time
import sys
from DSL import *
from rule_of_28_sketch import Rule_of_28_Player_PS
sys.path.insert(0,'..')
from players.random_player import RandomPlayer
from players.rule_of_28_player import Rule_of_28_Player
from players.rule_of_28_player_functional import Rule_of_28_Player_2
from players.glenn_player import Glenn_Player
from game import Board, Game


def play_match(p1, p2):
    game = Game(n_players = 2, dice_number = 4, dice_value = 6, column_range = [2,12],
                offset = 2, initial_height = 3)
    
    is_over = False
    who_won = None

    number_of_moves = 0
    current_player = game.player_turn
    while not is_over:
        moves = game.available_moves()
        if game.is_player_busted(moves):
            if current_player == 1:
                current_player = 2
            else:
                current_player = 1
            continue
        else:
            if game.player_turn == 1:
                chosen_play = p1.get_action(game)
            else:
                chosen_play = p2.get_action(game)
            if chosen_play == 'n':
                if current_player == 1:
                    current_player = 2
                else:
                    current_player = 1
            game.play(chosen_play)
            number_of_moves += 1
        who_won, is_over = game.is_finished()
        
        if is_over:
            return is_over, who_won
        
        if number_of_moves >= 300:
            print('Draw!')
            return False, None

def play_n_matches(p1, p2, n):
    
    p1_victories = 0
    p2_victories = 0
    
    for _ in range(n):
        # plays a match with br as player 1
        finished, who_won = play_match(p1, p2)
        
        if finished:
            if who_won == 1:
                p1_victories += 1
            else:
                p2_victories += 1
        
        # plays another match with br as player 2        
        finished, who_won = play_match(p2, p1)
        
        if finished:
            if who_won == 1:
                p2_victories += 1
            else:
                p1_victories += 1
    
    return p1_victories, p2_victories


if __name__ == "__main__":    
    
    program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
    program_decide_column = Argmax(Map(Function(Sum(Map(Function(Minus(Times(NumberAdvancedByAction(), VarScalarFromArray('move_value')), Times(VarScalar('marker'), IsNewNeutral()))), None))), VarList('actions')))
    
    program_yes_no1 = Sum(Map(Function(Times(NumberAdvancedThisRound(), VarScalarFromArray('progress_value'))), VarList('neutrals')))
#     program_yes_no2 = Sum(Map(Function(Plus(VarScalarFromArray('progress_value'), Times(NumberAdvancedThisRound(), NumberAdvancedThisRound()))), VarList('neutrals')))
    program_yes_no3 = Sum(Map(Function(Times(NumberAdvancedThisRound(), NumberAdvancedThisRound())), VarList('neutrals')))
    program_yes_no4 = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), NumberAdvancedThisRound()), VarScalarFromArray('progress_value'))), VarList('neutrals')))
    program_yes_no5 = Sum(Map(Function(Times(Times(NumberAdvancedThisRound(), NumberAdvancedThisRound()), VarScalarFromArray('progress_value'))), VarList('neutrals')))
    
    program_decide_column1 = Argmax(Map(Function(Sum(Map(Function(Times(NumberAdvancedByAction(), VarScalarFromArray('move_value'))), None))), VarList('actions')))
    program_decide_column2 = Argmax(Map(Function(Sum(VarList('neutrals'))), VarList('actions')))
    program_decide_column3 = Argmax(Map(Function(Sum(Map(Function(Plus(VarScalarFromArray('move_value'), NumberAdvancedByAction())), None))), VarList('actions')))
#     argmax(map((lambda x : sum(map((lambda x : (move_value + NumberAdvancedByAction)), None))), actions))
#     argmax(map((lambda x : sum(neutrals)), actions))
   
#     print(program_decide_column1.toString(), program_decide_column1.size)
    
    
    #sum(map((lambda x : (progress_value + (NumberAdvancedThisRound * NumberAdvancedThisRound))), neutrals))
#     p1 = Rule_of_28_Player()
#     p2 = Rule_of_28_Player()
#     p1 = Rule_of_28_Player_2()
    p1 = Glenn_Player()
#     p1 = Rule_of_28_Player_PS(program_yes_no, program_decide_column1)
    p2 = Rule_of_28_Player_PS(program_yes_no4, program_decide_column3)
#     p2 = Rule_of_28_Player_PS(program_yes_no5)
    
    victories1 = 0
    victories2 = 0
    
    start = time.time()
    
    victories1, victories2 = play_n_matches(p1, p2, 2000)
    
#     for _ in range(1000):
#         game = Game(n_players = 2, dice_number = 4, dice_value = 6, column_range = [2,12],
#                     offset = 2, initial_height = 3)
#         
#         is_over = False
#         who_won = None
#     
#         number_of_moves = 0
#         current_player = game.player_turn
#         while not is_over:
#             moves = game.available_moves()
#             if game.is_player_busted(moves):
#                 if current_player == 1:
#                     current_player = 2
#                 else:
#                     current_player = 1
#                 continue
#             else:
#                 if game.player_turn == 1:
#                     chosen_play = p1.get_action(game)
#                 else:
#                     chosen_play = p2.get_action(game)
#                 if chosen_play == 'n':
#                     if current_player == 1:
#                         current_player = 2
#                     else:
#                         current_player = 1
# #                 print('Chose: ', chosen_play)
# #                 game.print_board()
#                 game.play(chosen_play)
# #                 game.print_board()
#                 number_of_moves += 1
#                 
# #                 print()
#             who_won, is_over = game.is_finished()
#             
#             if number_of_moves >= 400:
#                 is_over = True
#                 who_won = -1
#                 print('No Winner!')
#                 
#         if who_won == 1:
#             victories1 += 1
#         if who_won == 2:
#             victories2 += 1
            
    end = time.time()
    print(victories1, victories2)
    print('Player 1: ', victories1 / (victories1 + victories2))
    print('Player 2: ', victories2 / (victories1 + victories2))
    print(end - start, ' seconds')
