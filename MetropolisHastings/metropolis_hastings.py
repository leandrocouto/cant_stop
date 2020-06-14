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
    	tree_max_nodes, temperature, temperature_dec):
        """
        - beta is a constant used in the MH score function.
        - data is a list of game state and action gotten from the oracle.
        - player_1 and player_2 are objects derived from Player used to 
          generate the dataset used as the oracle.
        - n_games is the number of games to be generated between the players.
        - n_iterations is the number of iteration in the main MH loop.
        - k is the number of samples from dataset to be evaluated.
        - tree is a parse tree implementation.
        - temperature is the temperature parameter for a simulated annealing 
          approach. This allows the search algorithm to explore more the 
          search space.
        - temperature_dec is the scalar that shows how much the current
          temperature will be decreased
        """
        self.beta = beta
        self.data = []
        self.player_1 = player_1
        self.player_2 = player_2
        self.n_games = n_games
        self.n_iterations = n_iterations
        self.k = k
        self.tree_max_nodes = tree_max_nodes
        self.temperature = temperature
        self.temperature_dec = temperature_dec
        self.tree = ParseTree(DSL('S'), self.tree_max_nodes)
        self.all_results = []
        self.passed_results = []

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

        # Main loop
        for i in range(self.n_iterations):
            start = time.time()
            # Make a copy of the tree for future mutation
            new_tree = pickle.loads(pickle.dumps(self.tree, -1))

            new_tree.mutate_tree()

            current_program = self.tree.generate_program()
            mutated_program = new_tree.generate_program()

            script_best_player = self.tree.generate_player(
                                                        current_program, 
                                                        self.k, 
                                                        self.n_iterations, 
                                                        self.tree_max_nodes
                                                        )
            script_mutated_player = new_tree.generate_player(
                                                        mutated_program, 
                                                        self.k, 
                                                        self.n_iterations, 
                                                        self.tree_max_nodes
                                                        )

            score_best, _, _, _ = self.calculate_score_function(
                                                        script_best_player, 
                                                        initial_data
                                                        )
            score_mutated, errors_mutated, errors_rate_mutated, \
            chosen_default_action_mutated = self.calculate_score_function(
                                                        script_mutated_player, 
                                                        initial_data
                                                        )
            n_errors = errors_mutated[0]
            n_errors_string_action = errors_mutated[1]
            n_errors_numeric_action = errors_mutated[2]
            total_errors_rate = errors_rate_mutated[0]
            total_string_errors_rate = errors_rate_mutated[1]
            total_numeric_errors_rate = errors_rate_mutated[2]

            # Update score given the SA parameters
            new_score_mutated = score_mutated**(1 / self.temperature)
            new_score_best = score_best**(1 / self.temperature)

            # Accept program only if new score is higher.
            accept = min(1, new_score_mutated/new_score_best)

            # Adjust the temperature accordingly.
            self.temperature = self.temperature_adjustment(self.temperature)

            self.all_results.append(
                                        (
                                            n_errors,
                                            n_errors_string_action,
                                            n_errors_numeric_action,
                                            total_errors_rate,
                                            total_string_errors_rate,
                                            total_numeric_errors_rate, 
                                            chosen_default_action_mutated
                                        )
                                    )
            # If the new synthesized program is better
            if accept == 1:
                self.tree = new_tree
                self.passed_results.append(
                                            (
                                                n_errors,
                                                n_errors_string_action,
                                                n_errors_numeric_action,
                                                total_errors_rate,
                                                total_string_errors_rate,
                                                total_numeric_errors_rate, 
                                                chosen_default_action_mutated
                                            )
                                        )
                print('Iteration -', i, 'New program accepted - Score = ', 
                        score_mutated,'Error rate = ', errors_rate_mutated, 
                        'n_errors = ', n_errors, 'Default action = ', 
                        chosen_default_action_mutated)

            elapsed_time = time.time() - start
            print('Iteration -', i, '- Elapsed time: ', elapsed_time)
            
        best_program = self.tree.generate_program()
        script_best_player = self.tree.generate_player(
                                                        best_program, 
                                                        self.k, 
                                                        self.n_iterations, 
                                                        self.tree_max_nodes
                                                        )

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/result' + '_' + str(self.k) + 'd_' \
               + str(self.n_iterations) + 'i_' + str(self.tree_max_nodes) + 'n/'
        if not os.path.exists(path):
            os.makedirs(path)
        script = Script(
                            best_program, 
                            self.k, 
                            self.n_iterations, 
                            self.tree_max_nodes
                        )
        script.saveFile(path)
        self.generate_graphs(path)

        full_run_elapsed_time = time.time() - full_run
        print('Full program elapsed time = ', full_run_elapsed_time)

        return best_program, script_best_player

    def temperature_adjustment(self, current_temperature):
        """ 
        Calculate the next temperature based on the current one. It uses the
        formula T_(k+1) = alpha * T_k. 
        More temperature schedules can be tested in the future.
        """

        return temperature_dec * current_temperature

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
            print('terminou jogo', i)
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
        errors, errors_rate, chosen_default_action = self.calculate_errors(
                                                                    program, 
                                                                    new_data
                                                                    )
        score = math.exp(-self.beta * errors_rate[0])
        return score, errors, errors_rate, chosen_default_action

    def calculate_errors(self, program, new_data):
        """ 
        Calculate how many times the program passed as parameter chose a 
        different action when compared to the oracle (actions from dataset).
        Return:
            - n_errors is the number of errors that the program chose when 
              compared to the actions chosen by the oracle.
            - n_errors_string_action is the number of errors that the program 
              chose when compared to the "string" actions chosen by the oracle.
            - n_errors_numeric_action is the number of errors that the program 
              chose when compared to the "numeric" actions chosen by the oracle.
            - chosen_default_action is the number of times the program chose
              the default action (this means it returned false for every if
              condition). Given in percentage related to the dataset.
        """
        n_errors = 0
        n_errors_string_action = 0
        n_errors_numeric_action = 0
        for i in range(len(new_data)):
            default_action_before = program.default_counter
            chosen_play = program.get_action(new_data[i][0])
            default_action_after = program.default_counter
            if chosen_play != new_data[i][1]:
                n_errors += 1

                if chosen_play in ['y', 'n']:
                    n_errors_string_action += 1
                else:
                    n_errors_numeric_action += 1
            # If the program chose the default action, flag it as a miss to
            # force the program to synthesize better if-conditions
            elif default_action_before != default_action_after:
                n_errors += 1

                if chosen_play in ['y', 'n']:
                    n_errors_string_action += 1
                else:
                    n_errors_numeric_action += 1
        total_errors_rate = n_errors / len(new_data)
        total_string_errors_rate = n_errors_string_action / len(new_data)
        total_numeric_errors_rate = n_errors_numeric_action / len(new_data)
        chosen_default_action = program.default_counter / len(new_data)
        errors = (n_errors, n_errors_string_action, n_errors_numeric_action)
        errors_rate = (
                        total_errors_rate, 
                        total_string_errors_rate,
                        total_numeric_errors_rate
                        )
        return errors, errors_rate, chosen_default_action

    def sample_from_data(self, k):
        """ Sample k instances from oracle data for evaluation. """

        index_list = sample(range(len(self.data)), len(self.data) - k)
        new_data = np.delete(self.data, index_list, 0)

        return new_data

    def generate_graphs(self, path):
        """ Generate graphs related to the current MH iteration. """

        suffix = str(self.k) + "d_" + str(self.n_iterations) + "i_" \
                + str(self.tree_max_nodes) + "n"

        # Number of errors -- all results
        x = [i for i in range(len(self.all_results))]
        y = [self.all_results[i][0] for i in range(len(self.all_results))]
        a = [i for i in range(len(self.all_results))]
        b = [self.all_results[i][1] for i in range(len(self.all_results))]
        m = [i for i in range(len(self.all_results))]
        n = [self.all_results[i][2] for i in range(len(self.all_results))]
        plt.plot(x, y, 'r', label="Total errors")
        plt.plot(a, b, 'g', label="String errors")
        plt.plot(m, n, 'b', label="Numeric errors")
        plt.legend(loc="best")
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Number of errors")
        plt.suptitle("Number of errors for all MH iterations - " + str(self.k) + " data")
        plt.savefig(path + "all_number_errors_" + suffix)
        plt.close()
        # Error rate -- all results
        x = [i for i in range(len(self.all_results))]
        y = [self.all_results[i][3] for i in range(len(self.all_results))]
        a = [i for i in range(len(self.all_results))]
        b = [self.all_results[i][4] for i in range(len(self.all_results))]
        m = [i for i in range(len(self.all_results))]
        n = [self.all_results[i][5] for i in range(len(self.all_results))]
        plt.plot(x, y, 'r', label="Total errors")
        plt.plot(a, b, 'g', label="String errors")
        plt.plot(m, n, 'b', label="Numeric errors")
        plt.legend(loc="best")
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Error rate (%)")
        plt.suptitle("Error percentage for all MH iterations - " + str(self.k) + " data")
        plt.savefig(path + "all_errors_rate_" + suffix)
        plt.close()
        # Chosen default action - all results
        x_axis = [i for i in range(len(self.all_results))]
        y_axis = [self.all_results[i][6] for i in range(len(self.all_results))]
        plt.plot(x_axis, y_axis)
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Rate on choosing the default action")
        plt.suptitle("Rate of how many times the program chose the default action - " + str(self.k) + " data")
        plt.savefig(path + "all_results_default_" + suffix)
        plt.close()

        # Number of errors -- passed results
        x = [i for i in range(len(self.passed_results))]
        y = [self.passed_results[i][0] for i in range(len(self.passed_results))]
        a = [i for i in range(len(self.passed_results))]
        b = [self.passed_results[i][1] for i in range(len(self.passed_results))]
        m = [i for i in range(len(self.passed_results))]
        n = [self.passed_results[i][2] for i in range(len(self.passed_results))]
        plt.plot(x, y, 'r', label="Total errors")
        plt.plot(a, b, 'g', label="String errors")
        plt.plot(m, n, 'b', label="Numeric errors")
        plt.legend(loc="best")
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Number of errors")
        plt.suptitle("Number of errors for only successful MH iterations - " + str(self.k) + " data")
        plt.savefig(path + "passed_number_errors_" + suffix)
        plt.close()
        # Error rate -- passed results
        x = [i for i in range(len(self.passed_results))]
        y = [self.passed_results[i][3] for i in range(len(self.passed_results))]
        a = [i for i in range(len(self.passed_results))]
        b = [self.passed_results[i][4] for i in range(len(self.passed_results))]
        m = [i for i in range(len(self.passed_results))]
        n = [self.passed_results[i][5] for i in range(len(self.passed_results))]
        plt.plot(x, y, 'r', label="Total errors")
        plt.plot(a, b, 'g', label="String errors")
        plt.plot(m, n, 'b', label="Numeric errors")
        plt.legend(loc="best")
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Error rate (%)")
        plt.suptitle("Error percentage for only successful MH iterations - " + str(self.k) + " data")
        plt.savefig(path + "passed_errors_rate_" + suffix)
        plt.close()
        # Chosen default action - passed results
        x_axis = [i for i in range(len(self.passed_results))]
        y_axis = [self.passed_results[i][6] for i in range(len(self.passed_results))]
        plt.plot(x_axis, y_axis)
        plt.xlabel("Metropolis Hastings iterations")
        plt.ylabel("Rate on choosing the default action")
        plt.suptitle("Rate of how many times the program chose the default action - " + str(self.k) + " data")
        plt.savefig(path + "passed_results_default_" + suffix)
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
    n_games = 1000
    iterations = 302 
    k = -1
    tree_max_nodes = 304
    # Simulated annealing parameters
    # If no SA is to be used, set both parameters to 1.
    temperature = 1
    temperature_dec = 0.98

    MH = MetropolisHastings(
                                beta, 
                                player_1, 
                                player_2, 
                                n_games, 
                                iterations, 
                                k, 
                                tree_max_nodes,
                                temperature,
                                temperature_dec
                            )
    best_program, script_best_player = MH.run()