import sys
sys.path.insert(0,'..')
import math
import copy
from game import Game
from play_game_template import play_single_game
from players.vanilla_uct_player import Vanilla_UCT
from players.uct_player import UCTPlayer
from players.random_player import RandomPlayer
from MetropolisHastings.parse_tree import ParseTree, Node
from MetropolisHastings.DSL import DSL
from Script import Script
import time
import pickle
import os.path
from random import sample
import numpy as np
import matplotlib.pyplot as plt

class MetropolisHastings:
    def __init__(self, beta, player_1, player_2, n_games, n_iterations, k,
    	tree_max_nodes):
        """
        - beta is a constant used in the MH score function.
        - data is a list of game state and action gotten from the oracle.
        - player_1 and player_2 are objects derived from Player used to 
          generate the dataset used as the oracle.
        - n_games is the number of games to be generated between the players.
        - n_iterations is the number of iteration in the main MH loop.
        - k is the number of samples from dataset to be evaluated.
        - tree is a parse tree implementation.
        """
        self.beta = beta
        self.data = []
        self.player_1 = player_1
        self.player_2 = player_2
        self.n_games = n_games
        self.n_iterations = n_iterations
        self.k = k
        self.tree_max_nodes = tree_max_nodes
        self.tree = ParseTree(DSL('S'), self.tree_max_nodes)
        self.all_results = []
        self.passed_results = []
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
        full_run = time.time()
        # Sample k data instances to be used in the evaluation
        if k == -1:
            self.k = len(self.data)
        initial_data = self.sample_from_data(self.k)

        self.tree.build_tree(self.tree.root)
        #current_best_program = self.tree.generate_random_program()

        # Main loop
        for i in range(self.n_iterations):
            start = time.time()
            # Make a copy of the tree for future mutation
            new_tree = pickle.loads(pickle.dumps(self.tree, -1))

            new_tree.mutate_tree()

            current_program = self.tree.generate_program()
            mutated_program = new_tree.generate_program()

            script_best_player = self.tree.generate_player(current_program, self.k, self.n_iterations)
            script_mutated_player = new_tree.generate_player(mutated_program, self.k, self.n_iterations)

            score_best, n_errors_best, errors_rate_best, _ = \
                        self.calculate_score_function(
                                                        script_best_player, 
                                                        initial_data
                                                    )
            score_mutated, n_errors_mutated, errors_rate_mutated, \
            chosen_default_action_mutated = self.calculate_score_function(
                                                        script_mutated_player, 
                                                        initial_data
                                                        )
            accept = min(1, score_mutated/score_best)
            self.all_results.append(
                                        (errors_rate_mutated, 
                                        n_errors_mutated, 
                                        chosen_default_action_mutated)
                                    )
            if accept == 1:
                self.tree = new_tree
                self.passed_results.append(
                                            (errors_rate_mutated, 
                                            n_errors_mutated, 
                                            chosen_default_action_mutated)
                                        )
                print('Iteration -', i, 'New program accepted - Score = ', score_mutated,'Error rate = ', errors_rate_mutated, 'n_errors = ', n_errors_mutated, 'Default action = ', chosen_default_action_mutated)
                #print('programa = ', current_best_program)
            elapsed_time = time.time() - start
            print('Iteration -', i, '- Elapsed time: ', elapsed_time)
            
        best_program = self.tree.generate_program()
        script_best_player = self.tree.generate_player(best_program, self.k, self.n_iterations)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/result' + '_' + str(self.k) + '_' + str(self.n_iterations) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        script = Script(best_program, self.k, self.n_iterations)
        script.saveFile(path)
        self.generate_graphs(path)

        full_run_elapsed_time = time.time() - full_run
        print('Full program elapsed time = ', full_run_elapsed_time)

        return best_program, script_best_player

    def generate_oracle_data(self):
        """ Generate data by playing games between player_1 and player_2. """
        for i in range(self.n_games):
            game = Game(self.n_players, self.dice_number, self.dice_value, 
                        self.column_range, self.offset, self.initial_height
                        )
            single_game_data, _ = self.simplified_play_single_game(
                                                        self.player_1, 
                                                        self.player_2, 
                                                        game, 
                                                        self.max_game_length
                                                        )
            # Append game data to file
            with open('dataset', 'ab') as f:
                for data in single_game_data:
                    pickle.dump(data, f)

    def calculate_score_function(self, program, new_data):
        """ 
        Score function that calculates who the program passed as parameter 
        "imitates" the actions taken by the oracle in the saved dataset.
        Return this program's score.
        """
        n_errors, errors_rate, chosen_default_action = self.calculate_errors(
                                                                    program, 
                                                                    new_data
                                                                    )
        score = math.exp(-self.beta * errors_rate)
        return score, n_errors, errors_rate, chosen_default_action

    def calculate_errors(self, program, new_data):
        """ 
        Calculate how many times the program passed as parameter chose a 
        different action when compared to the oracle (actions from dataset).
        Return:
            - n_errors is the number of errors that the program chose when 
              compared to the actions chosen by the oracle.
            - chosen_default_action is the number of times the program chose
              the default action (this means it returned false for every if
              condition). Given in percentage related to the dataset.
        """
        n_errors = 0
        for i in range(len(new_data)):
            chosen_play = program.get_action(new_data[i][0])
            if chosen_play != new_data[i][1]:
                n_errors += 1
        errors_rate = n_errors / len(new_data)
        chosen_default_action = program.default_counter / len(new_data)
        return n_errors, errors_rate, chosen_default_action

    def sample_from_data(self, k):
        """ Sample k instances from oracle data for evaluation. """

        index_list = sample(range(len(self.data)), len(self.data) - k)
        new_data = np.delete(self.data, index_list, 0)

        return new_data

    def generate_graphs(self, path):
        
        # Error rate -- all results
        x_axis = [i for i in range(len(self.all_results))]
        y_axis = [self.all_results[i][0] for i in range(len(self.all_results))]
        plt.plot(x_axis, y_axis)
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Error rate (%)")
        plt.suptitle("Error percentage for all MH iterations - " + str(self.k) + " data")
        plt.savefig(path + "all_results_percentage_" + str(self.k) + "_" + str(self.n_iterations))
        plt.close()

        # Error rate - passed results
        x_axis = [i for i in range(len(self.passed_results))]
        y_axis = [self.passed_results[i][0] for i in range(len(self.passed_results))]
        plt.plot(x_axis, y_axis)
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Error rate (%)")
        plt.suptitle("Error percentage for only successful MH iterations - " + str(self.k) + " data")
        plt.savefig(path + "passed_results_percentage_" + str(self.k) + "_" +  str(self.n_iterations))
        plt.close()

        # Chosen default action - all results
        x_axis = [i for i in range(len(self.all_results))]
        y_axis = [self.all_results[i][2] for i in range(len(self.all_results))]
        plt.plot(x_axis, y_axis)
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Rate on choosing the default action")
        plt.suptitle("Rate of how many times the program chose the default action - " + str(self.k) + " data")
        plt.savefig(path + "all_results_default_" + str(self.k) + "_" +  str(self.n_iterations))
        plt.close()

        # Chosen default action - passed results
        x_axis = [i for i in range(len(self.passed_results))]
        y_axis = [self.passed_results[i][2] for i in range(len(self.passed_results))]
        plt.plot(x_axis, y_axis)
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Rate on choosing the default action")
        plt.suptitle("Rate of how many times the program chose the default action - " + str(self.k) + " data")
        plt.savefig(path + "passed_results_default_" + str(self.k) + "_" +  str(self.n_iterations))
        plt.close()


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
                who_won, is_over = game.is_finished()

        return single_game_data, who_won


if __name__ == "__main__":
    player_1 = Vanilla_UCT(c = 1, n_simulations = 50)
    player_2 = Vanilla_UCT(c = 1, n_simulations = 50)
    random_player = RandomPlayer()
    beta = 0.5
    n_games = 200
    iterations = 1000
    k = -1
    tree_max_nodes = 300
    MH = MetropolisHastings(beta, player_1, player_2, n_games, iterations, k, tree_max_nodes)
    best_program, script_best_player = MH.run()