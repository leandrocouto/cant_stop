from game import Game
import time
from players.vanilla_uct_player import Vanilla_UCT
from players.alphazero_player import AlphaZeroPlayer
from players.uct_player import UCTPlayer
from statistics import Statistic

class Experiment:

    def __init__(self, n_players, dice_number, dice_value, column_range,
                    offset, initial_height, max_game_length):
        self.n_players = n_players
        self.dice_number = dice_number
        self.dice_value = dice_value
        self.column_range = column_range 
        self.offset = offset
        self.initial_height = initial_height
        self.max_game_length = max_game_length

    def play_single_game(self, player1, player2):
        """
        Play a single game between self.player1 and self.player2.
        Returns the data of the game in a NN channel format and who won.
        If the game reaches the maximum number of iterations, it returns
        0 representing no victories for any of the players. 
        Return an empty list if none of the players are instances of AlphaZeroPlayer
        or the data collected from the game otherwise.
        Return 1 if self.player1 won and 2 if self.player2 won.
        """
        data_of_a_game = []
        game = Game(self.n_players, self.dice_number, self.dice_value, self.column_range,
                    self.offset, self.initial_height)
        is_over = False
        rounds = 0
        while not is_over:
            rounds += 1
            # Collecting data for later input to the NN if any of the players are
            # subclasses of AlphaZeroPlayer.
            if isinstance(player1, AlphaZeroPlayer):
                channel_valid = player1.valid_positions_channel(self.column_range, self.offset, self.initial_height)
                channel_finished_1, channel_finished_2 = player1.finished_columns_channels(game, channel_valid)
                channel_won_column_1, channel_won_column_2 = player1.player_won_column_channels(game, channel_valid)
                channel_turn = player1.player_turn_channel(game, channel_valid)
                list_of_channels = [channel_valid, channel_finished_1, channel_finished_2,
                                    channel_won_column_1, channel_won_column_2, channel_turn]
            elif isinstance(player2, AlphaZeroPlayer):
                channel_valid = player2.valid_positions_channel(self.column_range, self.offset, self.initial_height)
                channel_finished_1, channel_finished_2 = player2.finished_columns_channels(game, channel_valid)
                channel_won_column_1, channel_won_column_2 = player2.player_won_column_channels(game, channel_valid)
                channel_turn = player2.player_turn_channel(game, channel_valid)
                list_of_channels = [channel_valid, channel_finished_1, channel_finished_2,
                                    channel_won_column_1, channel_won_column_2, channel_turn]
            moves = game.available_moves()
            if game.is_player_busted(moves):
                continue
            else:
                if game.player_turn == 1:
                    chosen_play = player1.get_action(game)
                    dist_probability = []
                    if isinstance(player1, UCTPlayer):
                        dist_probability = player1.get_dist_probability()
                else:
                    chosen_play = player2.get_action(game)
                    dist_probability = []
                    if isinstance(player2, UCTPlayer):
                        dist_probability = player2.get_dist_probability()
                if isinstance(player1, AlphaZeroPlayer) or isinstance(player2, AlphaZeroPlayer):
                    # Collecting data
                    current_play = [list_of_channels, dist_probability]
                    data_of_a_game.append(current_play)
                # Apply the chosen_play in the game
                game.play(chosen_play)
            # if the game has reached its max number of plays, end the game
            # and who_won receives 0, which means no players won.
            if rounds > self.max_game_length:
                who_won = 0
                is_over = True
            else:
                who_won, is_over = game.is_finished()
        return data_of_a_game, who_won

    def play_alphazero_iteration(self, old_model, current_model, use_UCT_playout, epochs, conv_number):
        """
        Both old_model and current_model are instances of AlphaZeroPlayer
        """

        # Stores data from net vs net in evaluation for later analysis.
        data_net_vs_net = []
        # Stores data from net vs uct in evaluation for later analysis.
        data_net_vs_uct = []
        #
        #
        # Main loop of the algorithm
        #
        #
        print()
        print('ALPHAZERO ITERATION')
        print()
        # 0 -> draw
        victory_0 = 0
        victory_1 = 0
        victory_2 = 0
        start = time.time()

        # dataset_for_network saves a list of info used as input for the network.
        # A list of: states of the game, distribution probability of the state
        # returned by the UCT and who won the game this state is in.
        dataset_for_network = []

        #
        #
        # Self-play
        #
        #

        for i in range(current_model.n_games):
            data_of_a_game, who_won = self.play_single_game(current_model, current_model)
            print('Self-play - GAME', i ,'OVER - PLAYER', who_won, 'WON')
            if who_won == 1:
                victory_1 += 1
            elif who_won == 2:
                victory_2 += 1
            else:
                victory_0 += 1
            # After the game is finished, we now know who won the game.
            # Therefore, save this info in all instances saved so far
            # for later use as input for the NN.
            for single_game in data_of_a_game:
                single_game.append(who_won)
            dataset_for_network.append(data_of_a_game)
        print('Self-play - Player 1 won', victory_1,'time(s).')
        print('Self-play - Player 2 won', victory_2,'time(s).')

        #
        #
        # Training
        #
        #

        print()
        print('TRAINING LOOP')
        print()

        x_train, y_train = current_model.transform_dataset_to_input(dataset_for_network)

        history_callback = current_model.network.fit(x_train, y_train, epochs = epochs, shuffle = True)

        # Saving data
        loss_history = sum(history_callback.history["loss"]) / len(history_callback.history["loss"])
        output_dist_loss_history = sum(history_callback.history["Output_Dist_loss"]) / \
                                    len(history_callback.history["Output_Dist_loss"])
        output_value_loss_history = sum(history_callback.history["Output_Value_loss"]) / \
                                    len(history_callback.history["Output_Value_loss"])
        dist_metric_history = sum(history_callback.history["Output_Dist_categorical_crossentropy"]) / \
                                len(history_callback.history["Output_Dist_categorical_crossentropy"])
        value_metric_history = sum(history_callback.history["Output_Value_mean_squared_error"]) /  \
                                len(history_callback.history["Output_Value_mean_squared_error"])

        #
        #    
        # Model evaluation
        #
        #
        
        print()
        print('MODEL EVALUATION - Network vs. Old Network')
        print()

        # The current network faces the previous one.
        # If it does not win victory_rate % it completely
        # discards the current network.

        victory_1_eval_net = 0
        victory_2_eval_net = 0

        for i in range(current_model.n_games_evaluate):
            data_of_a_game, who_won = self.play_single_game(current_model, old_model)
            if who_won == 1:
                victory_1_eval_net += 1
            else:
                victory_2_eval_net += 1

        data_net_vs_net.append((victory_1_eval_net, victory_2_eval_net))

        necessary_won_games = (current_model.victory_rate * current_model.n_games_evaluate) / 100

        if victory_1_eval_net <= necessary_won_games:
            print('New model is worse and won ', victory_1_eval_net)
        else:
            # Overwrites the old model with the current one.
            old_model.network.set_weights(current_model.network.get_weights())
            print('New model is better and won ', victory_1_eval_net)

            # The new model is better. Therefore, evaluate it against
            # vanilla UCT and store the data for later analysis.

            dataset_for_eval = []
            # Saving data
            loss_eval = []
            dist_loss_eval = []
            value_loss_eval = []
            dist_metric_eval = []
            value_metric_eval = []
            victory_1_eval = 0
            victory_2_eval = 0

            print()
            print('MODEL EVALUATION - Network vs. UCT')
            print()

            uct_player = Vanilla_UCT(current_model.c, current_model.n_simulations)

            for i in range(current_model.n_games_evaluate):
                data_of_a_game_eval, who_won = self.play_single_game(current_model, uct_player)
                print('Net vs UCT - GAME', i ,'OVER - PLAYER', who_won, 'WON')
                if who_won == 1:
                    victory_1_eval += 1
                else:
                    victory_2_eval += 1
                # After the game is finished, we now know who won the game.
                # Therefore, save this info in all instances saved so far
                # for later use as input for the NN.
                for single_game in data_of_a_game_eval:
                    single_game.append(who_won)
                dataset_for_eval.append(data_of_a_game_eval)
                print('Net vs UCT - Network won', victory_1_eval,'time(s).')
                print('Net vs UCT - UCT won', victory_2_eval,'time(s).')

                x_train_eval, y_train_eval = current_model.transform_dataset_to_input(dataset_for_eval)

                results = current_model.network.evaluate(x_train_eval, y_train_eval)
                # Saving data
                loss_eval.append(results[0])
                dist_loss_eval.append(results[1])
                value_loss_eval.append(results[2])
                dist_metric_eval.append(results[3])
                value_metric_eval.append(results[4])
                dataset_for_eval = []

            loss_eval = sum(loss_eval) / len(loss_eval)
            dist_loss_eval = sum(dist_loss_eval) / len(dist_loss_eval)
            value_loss_eval = sum(value_loss_eval) / len(value_loss_eval)
            dist_metric_eval = sum(dist_metric_eval) / len(dist_metric_eval)
            value_metric_eval = sum(value_metric_eval) / len(value_metric_eval)

            

            # Saving data
            data_net_vs_uct.append((loss_history, output_dist_loss_history, output_value_loss_history, 
                dist_metric_history, value_metric_history, victory_1, victory_2, loss_eval, dist_loss_eval, 
                value_loss_eval, dist_metric_eval, value_metric_eval, victory_1_eval, victory_2_eval))

            elapsed_time = time.time() - start
            print('Time elapsed: ', elapsed_time)

            stats = Statistic(data_net_vs_net, data_net_vs_uct, n_simulations = current_model.n_simulations,
                     n_games = current_model.n_games, alphazero_iterations = current_model.alphazero_iterations, 
                     use_UCT_playout = use_UCT_playout, conv_number = conv_number)

            return stats, old_model, current_model

        # If the new model is worse than the previous one,
        # then the stats and the new trained network is discarded.
        return [], old_model, old_model
               	
