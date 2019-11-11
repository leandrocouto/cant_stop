from collections import defaultdict
import random
from game import Board, Game
from uct import MCTS
import time
import numpy as np
import copy

class Config:
    """ General configuration class for the game board, UCT and NN """
    def __init__(self, c, n_simulations, n_games, n_players, dice_number,
                    dice_value, column_range, offset, initial_height):
        """
        - c is the constant the balance exploration and exploitation.
        - n_simulations is the number of simulations made in the UCT algorithm.
        - n_games is the number of games played in the self-play scheme.
        - n_players is the number of players (At the moment, only 2 is possible).
        - dice_number is the number of dice used in the Can't Stop game.
        - dice_value is the number of sides of a single die.
        - column_range is a list denoting the range of the board game columns.
        - offset is the height difference between columns.
        - initial_height is the height of the columns at the border of the board.
        """
        self.c = c
        self.n_simulations = n_simulations
        self.n_games = n_games
        self.n_players = n_players
        self.dice_number = dice_number
        self.dice_value = dice_value
        self.column_range = column_range 
        self.offset = offset
        self.initial_height = initial_height

def main():
    victory_1 = 0
    victory_2 = 0
    config = Config(c =1, n_simulations = 100, n_games = 10, n_players = 2, 
                    dice_number = 4, dice_value = 3, column_range = [2,6], 
                    offset = 2, initial_height = 1)
    start = time.time()
    for i in range(config.n_games):
        game = Game(config)
        is_over = False
        uct = MCTS(config)
        print('Game', i, 'has started.')
        while not is_over:
            #print('player_id = ', game.player_turn)
            game.board_game.print_board(game.player_won_column)
            #print('Dice: ', game.current_roll)
            moves = game.available_moves()
            #print('Available plays: ', moves)
            if game.is_player_busted(moves):
                #print('\nPlayer busted!\n')
                continue
            else:
                if game.player_turn == 1:
                    chosen_play, root = uct.run_mcts(game)
                else:
                    chosen_play, root = uct.run_mcts(game)#chosen_play = random.choice(moves)
                #print('Chosen play:', chosen_play)
                game.play(chosen_play)
            #print('Finished columns: ', game.finished_columns)
            #print('Player won column: ',game.player_won_column)
            #game.board_game.print_board(game.player_won_column)
            who_won, is_over = game.is_finished()
        #print()
        #print('GAME', i ,'OVER - PLAYER', who_won, 'WON')
        if who_won == 1:
            victory_1 += 1
        else:
            victory_2 += 1
        #print()
        #game.board_game.print_board(game.player_won_column)
        #print('Finished columns: ', game.finished_columns)
    print('Player 1 won', victory_1,'time(s).')
    print('Player 2 won', victory_2,'time(s).')
    end = time.time()
    print('Time elapsed:', end - start)

if __name__ == "__main__":
    main()