from collections import defaultdict
import random
from game import Board, Game

if __name__ == "__main__":
    n_players = 2
    game = Game(n_players)
    #game.board_game.board[12][0].markers.append(1)
    game.board_game.board[7][10].markers.append(1)
    player_id = 1
    while True:
        roll = game.roll_dice()
        print('Dice: ', roll)
        moves = game.available_moves(roll)
        print('Finished columns: ', game.finished_columns)
        print('Player won column: ',game.player_won_column)
        print('Available plays: ', moves)
        if game.is_player_busted(moves):
            print('Player busted')
            game.erase_neutral_markers()
            game.finished_columns.clear()
            game.player_won_column.clear()
            if player_id == n_players:
                player_id = 1
            else:
                player_id += 1
            print('player_id = ', player_id)
            continue
        chosen_play = int(input('Choose play: '))
        game.play(player_id, moves[chosen_play])
        game.board_game.print_board()
        print('Após play.')
        continue_to_play = input('Continue? (y/n)') 
        if continue_to_play != 'y' and continue_to_play != 'Y':
            game.transform_neutral_markers(player_id)
            if player_id == n_players:
                player_id = 1
            else:
                player_id += 1 
            print('player_id = ', player_id)
            continue
    game.board_game.print_board()
