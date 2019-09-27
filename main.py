from collections import defaultdict
import random
from game import Board, Game
from uct import MCTS

if __name__ == "__main__":
    n_players = 2
    game = Game(n_players = n_players)
    is_over = False
    uct = MCTS(1, 1000)
    
    while not is_over:
        print('player_id = ', game.player_turn)
        game.board_game.print_board(game.player_won_column)
        print('Dice: ', game.current_roll)
        moves = game.available_moves()
        print('Available plays: ', moves)
        if game.is_player_busted(moves):
            print('Player busted!')
            continue
        else:
            print('antes run mcts')
            chosen_play, _ = uct.run_mcts(game)
            print('depois run mcts')
            print('Chosen play:', chosen_play)
            game.play(chosen_play)
        print('Finished columns: ', game.finished_columns)
        print('Player won column: ',game.player_won_column)
        game.board_game.print_board(game.player_won_column)
        who_won, is_over = game.is_finished()
    print()
    print('GAME OVER - PLAYER', who_won, 'WON')
    print()
    game.board_game.print_board(game.player_won_column)
    print('Finished columns: ', game.finished_columns)