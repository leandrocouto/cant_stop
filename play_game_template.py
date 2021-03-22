import pickle
import math
import time
import re
from game import Game

def simplified_play_single_game(player1, player2, game, max_game_length):
    """
    Play a single game between player1 and player2.
    Return 1 if player1 won, 2 if player2 won or if the game reached the max
    number of iterations, it returns 0 representing a draw.
    In this method, it is not stored what the other player played in previous
    rounds (this info is used in UCTPlayer players in order to maintain the 
    UCT tree between plays).
    """

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

        # if the game has reached its max number of plays, end the game
        # and who_won receives 0, which means no players won.

        if rounds > max_game_length:
            who_won = 0
            is_over = True
        else:
            who_won, is_over = game.is_finished()

    return who_won


def play_solitaire_single_game(player, game, max_game_length):
    """ Play a solitaire version of the Can't Stop game. """

    is_over = False
    i = -1
    # Game loop
    while not is_over:
        i += 1
        moves = game.available_moves()
        if game.is_player_busted(moves):
            continue
        else:
            chosen_play = player.get_action(game)
            # Apply the chosen_play in the game
            game.play(chosen_play)

        # if the game has reached its max number of plays, end the game
        if game.n_rounds > max_game_length:
            is_over = True
        else:
            is_over = game.is_finished()

    return game.n_rounds