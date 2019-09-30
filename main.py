from collections import defaultdict
import random
from game import Board, Game
from uct import MCTS
import time

def main():
    victory_1 = 0
    victory_2 = 0
    c = 1
    n_simulations = 10
    n_games = 10
    n_players = 2
    start = time.time()
    for i in range(n_games):
        game = Game(n_players = n_players)
        is_over = False
        uct = MCTS(c, n_simulations)
        print('Game', i, 'has started.')
        while not is_over:
            #print('player_id = ', game.player_turn)
            #game.board_game.print_board(game.player_won_column)
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
                    chosen_play = random.choice(moves)
                #print('Chosen play:', chosen_play)
                game.play(chosen_play)
            #print('Finished columns: ', game.finished_columns)
            #print('Player won column: ',game.player_won_column)
            #game.board_game.print_board(game.player_won_column)
            who_won, is_over = game.is_finished()
        #print()
        print('GAME', i ,'OVER - PLAYER', who_won, 'WON')
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