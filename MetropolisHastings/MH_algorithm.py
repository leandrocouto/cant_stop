import sys
sys.path.insert(0,'..')
import math
import copy
from game import Game
from play_game_template import play_single_game
from players.vanilla_uct_player import Vanilla_UCT
from players.uct_player import UCTPlayer
from players.random_player import RandomPlayer
from MetropolisHastings.MH_tree import DSL, DSLTree, Node
import time
import pickle
import os.path

class MetropolisHastings:
    def __init__(self, beta, player_1, player_2, n_games, n_iterations):
        """
        - beta is a constant used in the MH score function.
        - data is a list of game state and action gotten from the oracle.
        - player_1 and player_2 are objects derived from Player used to 
          generate the dataset used as the oracle.
        - n_games is the number of games to be generated between the players.
        - n_iterations is the number of iteration in the main MH loop.
        """
        self.beta = beta
        self.data = []
        self.player_1 = player_1
        self.player_2 = player_2
        self.n_games = n_games
        self.n_iterations = n_iterations
        '''
        # Toy version
        self.column_range = [2,6]
        self.offset = 2
        self.initial_height = 1
        self.n_players = 2
        self.dice_number = 4
        self.dice_value = 3
        self.max_game_length = 50
        '''
        # Original version
        self.column_range = [2,12]
        self.offset = 2
        self.initial_height = 2 
        self.n_players = 2
        self.dice_number = 4
        self.dice_value = 6 
        self.max_game_length = 500
        
    def run(self):
        """ Main routine of the MH algorithm. """
        dsl = DSL()
        tree = DSLTree(Node('S', ''), dsl)
        tree.build_tree()
        current_best_program = tree.generate_random_program()

        # Check if there's already data available. If not, generate it.
        if not os.path.isfile('dataset'):
            self.generate_oracle_data()
        # If there is, read from it.
        with open('dataset', "rb") as f:
            while True:
                try:
                    self.data.append(pickle.load(f))
                except EOFError:
                    break
        
        # Main loop
        for i in range(self.n_iterations):
            
            new_tree = copy.deepcopy(tree)
            mutated_program = new_tree.generate_mutated_program(current_best_program)

            script_best_player = tree.generate_player(current_best_program)
            script_mutated_player = new_tree.generate_player(mutated_program)

            score_best, n_errors_best, errors_rate_best, v = \
                        self.calculate_score_function(script_best_player)
            score_mutated, n_errors_mutated, errors_rate_mutated, v2 = \
                        self.calculate_score_function(script_mutated_player)

            accept = min(1, score_mutated/score_best)
            if accept == 1:
                current_best_program = mutated_program
                tree = new_tree
            
        return current_best_program

    def generate_oracle_data(self):
        """ Generate data by playing games between player_1 and player_2. """
        for i in range(self.n_games):
            game = Game(self.n_players, self.dice_number, self.dice_value, 
                        self.column_range, self.offset, self.initial_height
                        )
            single_game_data = self.simplified_play_single_game(
                                                        self.player_1, 
                                                        self.player_2, 
                                                        game, 
                                                        self.max_game_length
                                                        )
            # Append game data to file
            with open('dataset', 'ab') as f:
                for data in single_game_data:
                    pickle.dump(data, f)

    def calculate_score_function(self, program):
        """ 
        Score function that calculates who the program passed as parameter 
        "imitates" the actions taken by the oracle in the saved dataset.

        """
        n_errors, errors_rate, v = self.calculate_errors(program)
        return math.exp(-self.beta * n_errors), n_errors, errors_rate, v

    def calculate_errors(self, program):
        """ 
        Calculate how many times the program passed as parameter chose a 
        different action when compared to the oracle (actions from dataset).
        """
        n_errors = 0
        v = 0
        for i in range(len(self.data)):
            chosen_play, value = program.get_action(self.data[i][0])
            if value == 1:
                v += 1
            if chosen_play != self.data[i][1]:
                n_errors += 1
        return n_errors,  n_errors / len(self.data), v


    def simplified_play_single_game(self, player_1, player_2, game, 
        max_game_length):
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

                single_game_data.append((game.clone(), chosen_play))
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
                _, is_over = game.is_finished()

        return single_game_data


if __name__ == "__main__":
    player_1 = Vanilla_UCT(c = 1, n_simulations = 50)
    player_2 = Vanilla_UCT(c = 1, n_simulations = 50)
    beta = 0.5
    n_games = 3
    iterations = 1000
    MH = MetropolisHastings(beta, player_1, player_2, n_games, iterations)
    for i in range(1):
        MH.data = []
        program = MH.run()
        print(program)
