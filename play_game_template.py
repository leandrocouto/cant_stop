from game import Game
from players.vanilla_uct_player import Vanilla_UCT
from players.uct_player import UCTPlayer
from players.glenn_player import Glenn_Player
from sketch import Sketch
import pickle
import math
import time
import re

def play_single_game(player1, player2, game, max_game_length):
    """
    Play a single game between player1 and player2.
    Return 1 if player1 won, 2 if player2 won or if the game reached the max
    number of iterations, it returns 0 representing a draw.
    """

    is_over = False
    rounds = 0
    # actions_taken actions in a row from a player. Used in UCTPlayer players. 
    # List of tuples (action taken, player turn, Game instance).
    # If players change turn, empty the list.
    actions_taken = []
    actions_from_player = 1

    # Game loop
    while not is_over:
        rounds += 1
        moves = game.available_moves()
        if game.is_player_busted(moves):
            actions_taken = []
            actions_from_player = game.player_turn
            continue
        else:
            # UCTPlayer players receives an extra parameter in order to
            # maintain the tree between plays whenever possible
            if game.player_turn == 1 and isinstance(player1, UCTPlayer):
                if actions_from_player == game.player_turn:
                    chosen_play = player1.get_action(game, [])
                else:
                    chosen_play = player1.get_action(game, actions_taken)
            elif game.player_turn == 1 and not isinstance(player1, UCTPlayer):
                    chosen_play = player1.get_action(game)
            elif game.player_turn == 2 and isinstance(player2, UCTPlayer):
                if actions_from_player == game.player_turn:
                    chosen_play = player2.get_action(game, [])
                else:
                    chosen_play = player2.get_action(game, actions_taken)
            elif game.player_turn == 2 and not isinstance(player2, UCTPlayer):
                    chosen_play = player2.get_action(game)

            # Needed because game.play() can automatically change 
            # the player_turn attribute.
            actual_player = game.player_turn
            
            # Clear the plays info so far if player_turn 
            # changed last iteration.
            if actions_from_player != actual_player:
                actions_taken = []
                actions_from_player = game.player_turn

            # Apply the chosen_play in the game
            game.play(chosen_play)

            # Save game history
            actions_taken.append((chosen_play, actual_player, game.clone()))

        # if the game has reached its max number of plays, end the game
        # and who_won receives 0, which means no players won.

        if rounds > max_game_length:
            who_won = 0
            is_over = True
        else:
            who_won, is_over = game.is_finished()

    return who_won

def play_single_game_parallel_one_script(program_string, program_column, 
    other_player, game, max_game_length, iteration, is_first):
    """
    Play a single game between player1 and player2.
    Return 1 if player1 won, 2 if player2 won or if the game reached the max
    number of iterations, it returns 0 representing a draw.
    """

    is_over = False
    rounds = 0

    if is_first:
        # Dynamically instantiation of the first player
        script_1 = Sketch(program_string, program_column, iteration, max_game_length)

        script_1_code = script_1.generate_text(iteration)
        exec(script_1_code)
        player1 = locals()[re.search("class (.*):", script_1_code).group(1).partition("(")[0]]()

        player2 = other_player
    else:
        # Dynamically instantiation of the first player
        script_2 = Sketch(program_string, program_column, iteration, max_game_length)

        script_2_code = script_2.generate_text(iteration)
        exec(script_2_code)
        player2 = locals()[re.search("class (.*):", script_2_code).group(1).partition("(")[0]]()

        player1 = other_player
    # actions_taken actions in a row from a player. Used in UCTPlayer players. 
    # List of tuples (action taken, player turn, Game instance).
    # If players change turn, empty the list.
    actions_taken = []
    actions_from_player = 1

    # Game loop
    while not is_over:
        rounds += 1
        moves = game.available_moves()
        if game.is_player_busted(moves):
            actions_taken = []
            actions_from_player = game.player_turn
            continue
        else:
            # UCTPlayer players receives an extra parameter in order to
            # maintain the tree between plays whenever possible
            if game.player_turn == 1 and isinstance(player1, UCTPlayer):
                if actions_from_player == game.player_turn:
                    chosen_play = player1.get_action(game, [])
                else:
                    chosen_play = player1.get_action(game, actions_taken)
            elif game.player_turn == 1 and not isinstance(player1, UCTPlayer):
                    chosen_play = player1.get_action(game)
            elif game.player_turn == 2 and isinstance(player2, UCTPlayer):
                if actions_from_player == game.player_turn:
                    chosen_play = player2.get_action(game, [])
                else:
                    chosen_play = player2.get_action(game, actions_taken)
            elif game.player_turn == 2 and not isinstance(player2, UCTPlayer):
                    chosen_play = player2.get_action(game)

            # Needed because game.play() can automatically change 
            # the player_turn attribute.
            actual_player = game.player_turn
            
            # Clear the plays info so far if player_turn 
            # changed last iteration.
            if actions_from_player != actual_player:
                actions_taken = []
                actions_from_player = game.player_turn

            # Apply the chosen_play in the game
            game.play(chosen_play)

            # Save game history
            actions_taken.append((chosen_play, actual_player, game.clone()))

        # if the game has reached its max number of plays, end the game
        # and who_won receives 0, which means no players won.

        if rounds > max_game_length:
            who_won = 0
            is_over = True
        else:
            who_won, is_over = game.is_finished()

    return who_won

def simplified_play_single_game_parallel_one_script(current_program_string, 
    current_program_column, other_player, game, max_game_length, iteration, 
    is_first):
    """
    Play a single game between player1 and player2 for parallel calls, that's
    why the dynamic instantiation happens here, because dynamic objects cannot
    be pickled and therefore cannot be passed as parameters for parallel calls.

    Assume only the first player is dynamic instantiated in this method.
    
    - is_first is a boolean telling if the dynamic instantiated script is to be
      played as first player or not.

    Return 1 if player1 won, 2 if player2 won or if the game reached the max
    number of iterations, it returns 0 representing a draw.
    In this method, it is not stored what the other player played in previous
    rounds (this info is used in UCTPlayer players in order to maintain the 
    UCT tree between plays).
    """

    is_over = False
    rounds = 0
    
    if is_first:
        # Dynamically instantiation of the first player
        script_1 = Sketch(current_program_string, current_program_column, iteration, max_game_length)

        script_1_code = script_1.generate_text(iteration)
        exec(script_1_code)
        player1 = locals()[re.search("class (.*):", script_1_code).group(1).partition("(")[0]]()

        player2 = other_player
    else:
        # Dynamically instantiation of the first player
        script_2 = Sketch(current_program_string, current_program_column, iteration, max_game_length)

        script_2_code = script_2.generate_text(iteration)
        exec(script_2_code)
        player2 = locals()[re.search("class (.*):", script_2_code).group(1).partition("(")[0]]()

        player1 = other_player

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

def simplified_play_single_game_parallel_two_scripts(first_program_string, 
    first_program_column, second_program_string, second_program_column, 
    game, max_game_length, iteration):
    """
    Play a single game between player1 and player2 for parallel calls, that's
    why the dynamic instantiation happens here, because dynamic objects cannot
    be pickled and therefore cannot be passed as parameters for parallel calls.


    Return 1 if player1 won, 2 if player2 won or if the game reached the max
    number of iterations, it returns 0 representing a draw.
    In this method, it is not stored what the other player played in previous
    rounds (this info is used in UCTPlayer players in order to maintain the 
    UCT tree between plays).
    """

    is_over = False
    rounds = 0
    
    # Dynamically instantiation of the first player
    script_1 = Sketch(first_program_string, first_program_column, iteration, max_game_length)

    script_1_code = script_1.generate_text(iteration)
    exec(script_1_code)
    player1 = locals()[re.search("class (.*):", script_1_code).group(1).partition("(")[0]]()

    # Dynamically instantiation of the second player
    script_2 = Sketch(second_program_string, second_program_column, iteration, max_game_length)

    script_2_code = script_2.generate_text(iteration)
    exec(script_2_code)
    player2 = locals()[re.search("class (.*):", script_2_code).group(1).partition("(")[0]]()


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