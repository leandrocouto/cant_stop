import sys
sys.path.insert(0,'..')
import math
import copy
from game import Game
from players.vanilla_uct_player import Vanilla_UCT
from players.uct_player import UCTPlayer
from players.random_player import RandomPlayer
import time
import pickle
import os.path

class HIGHLIGHTS:
    def __init__(self, player_1, player_2, n_games):
        self.data = []
        self.player_1 = player_1
        self.player_2 = player_2
        self.n_games = n_games

        # Original version
        self.column_range = [2,12]
        self.offset = 2
        self.initial_height = 2 
        self.n_players = 2
        self.dice_number = 4
        self.dice_value = 6 
        self.max_game_length = 500

    def sort_data_by_importance(self):

        with open('dataset_test', "rb") as f:
            while True:
                try:
                    self.data.append(pickle.load(f))
                except EOFError:
                    break

        self.data = sorted(self.data, key=lambda tup: tup[3], reverse = True)
        for d in self.data:
            print(d)
    def generate_oracle_data(self):
        """ Generate data by playing games between player_1 and player_2. """

        for i in range(self.n_games):
            game_run = time.time()
            game = Game(self.n_players, self.dice_number, self.dice_value, 
                        self.column_range, self.offset, self.initial_height
                        )
            single_game_data, _ = self.play_single_game(
                                                        self.player_1, 
                                                        self.player_2, 
                                                        game, 
                                                        self.max_game_length
                                                        )
            elapsed_time = time.time() - game_run
            print('Game', i, 'finished. Time elapsed:', elapsed_time)
            # Append game data to file
            with open('dataset_test', 'ab') as f:
                for data in single_game_data:
                    pickle.dump(data, f)

    def play_single_game(self, player_1, player_2, game, max_game_length):
        """ Play a single game between player_1 and player_2. """

        is_over = False
        rounds = 0
        # actions_taken actions in a row from a UCTPlayer player. 
        # List of tuples (action taken, player turn, Game instance).
        # If players change turn, empty the list.
        actions_taken = []
        actions_from_player = 1

        single_game_data = []

        # Game loop
        while not is_over:
            rounds += 1
            moves = game.available_moves()
            #print('available actions = ', moves)
            if game.is_player_busted(moves):
                actions_taken = []
                actions_from_player = game.player_turn
                continue
            else:
                # UCTPlayer players receives an extra parameter in order to
                # maintain the tree between plays whenever possible
                if game.player_turn == 1 and isinstance(player_1, UCTPlayer):
                    if actions_from_player == game.player_turn:
                        chosen_play = player_1.get_action(game, [])
                    else:
                        chosen_play = player_1.get_action(game, actions_taken)
                elif game.player_turn == 1 and not isinstance(player_1, UCTPlayer):
                        chosen_play = player_1.get_action(game)
                elif game.player_turn == 2 and isinstance(player_2, UCTPlayer):
                    if actions_from_player == game.player_turn:
                        chosen_play = player_2.get_action(game, [])
                    else:
                        chosen_play = player_2.get_action(game, actions_taken)
                elif game.player_turn == 2 and not isinstance(player_2, UCTPlayer):
                        chosen_play = player_2.get_action(game)

                if game.player_turn == 1:
                    #print('q_a p1 = ', player_1.q_a_root)
                    #print('dist = ', player_1.dist_probability)
                    q_a = dict(player_1.q_a_root)
                else:
                    #print('q_a p2 = ', player_2.q_a_root)
                    #print('dist = ', player_2.dist_probability)
                    q_a = dict(player_2.q_a_root)
                #print('chosen play = ', chosen_play)
                #print()
                if len(moves) > 1:
                    importance = q_a[max(q_a, key=q_a.get)] - q_a[min(q_a, key=q_a.get)]
                    #print('q_a = ', q_a)
                    #print('imp = ', importance)
                    single_game_data.append((game.clone(), chosen_play, q_a, importance))
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
                is_over = True
            else:
                who_won, is_over = game.is_finished()

        return single_game_data, who_won

if __name__ == "__main__":
    player_1 = Vanilla_UCT(c = 1, n_simulations = 500)
    player_2 = Vanilla_UCT(c = 1, n_simulations = 500)
    n_games = 940
    highlights = HIGHLIGHTS(player_1, player_2, n_games)
    highlights.generate_oracle_data()
    #highlights.sort_data_by_importance()