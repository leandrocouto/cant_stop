from collections import defaultdict
import random
from game import Board, Game
from uct import MCTS
import time

if __name__ == "__main__":
    victory_1 = 0
    victory_2 = 0
    start = time.time()
    for i in range(50):
        n_players = 2
        game = Game(n_players = n_players)
        is_over = False
        uct = MCTS(c=1, n_simulations=50)
        print('Jogo', i, 'começou.')
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
                #print('antes run mcts')
                if game.player_turn == 1:
                    chosen_play, root = uct.run_mcts(game)
                else:
                    chosen_play = random.choice(moves)
                #game = root.state
                #print('depois run mcts')
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
    print('Player 1 ganhou', victory_1,'vezes.')
    print('Player 2 ganhou', victory_2,'vezes.')
    end = time.time()
    print('time elapsed: ', end - start)