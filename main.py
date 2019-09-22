from collections import defaultdict
import random
from game import Board, Game
from uct import MCTS

if __name__ == "__main__":
    n_players = 2
    game = Game(n_players)
    while not game.is_finished():
        print('player_id = ', game.player_turn)
        game.board_game.print_board(game.player_won_column)
        roll = game.roll_dice()
        print('Dice: ', roll)
        moves = game.available_moves(roll)
        print('Available plays: ', moves)
        if moves == ['y','n']:
            chosen_play = random.choice(moves)
            print('Chosen play y/n:', chosen_play)
            game.play(chosen_play)
        else:
            if game.is_player_busted(moves):
                print('Player busted')
                continue
            else:
                chosen_play = random.choice(moves)
                print('Chosen play:', chosen_play)
                game.play(chosen_play)
        print('Finished columns: ', game.finished_columns)
        print('Player won column: ',game.player_won_column)
        game.board_game.print_board(game.player_won_column)
    print()
    print('GAME OVER')
    print()
    game.board_game.print_board(game.player_won_column)
    print('Finished columns: ', game.finished_columns)
