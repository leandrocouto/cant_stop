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

    def play_alphazero_iteration(self, current_model, old_model, UCTs_eval, use_UCT_playout, epochs, conv_number):
        """
        Both old_model and current_model are instances of AlphaZeroPlayer
        """

        file_name = str(current_model.n_simulations)+'_'+str(current_model.n_games) \
                + '_' + str(current_model.alphazero_iterations) + '_' + str(conv_number) + \
                '_' + str(use_UCT_playout) + '.txt'
        # Stores data from net vs net in rtaining for later analysis.
        data_net_vs_net_training = []
        # Stores data from net vs net in evaluation for later analysis.
        data_net_vs_net_eval = []
        # Stores data from net vs uct in evaluation for later analysis.
        data_net_vs_uct = []
        #
        #
        # Main loop of the algorithm
        #
        #

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
            #print('Self-play - GAME', i ,'OVER - PLAYER', who_won, 'WON')
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

            # Ties (infinite games) are not trained
            if who_won != 0:
                dataset_for_network.append(data_of_a_game)

        with open(file_name, 'a') as f:
            print('Self-play - Player 1 won', victory_1,'time(s).', file=f)
            print('Self-play - Player 2 won', victory_2,'time(s).', file=f)
            print('Self-play - Ties:', victory_0, file=f)

        #
        #
        # Training
        #
        #
        with open(file_name, 'a') as f:
            print('TRAINING LOOP', file=f)

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

        data_net_vs_net_training.append((loss_history, output_dist_loss_history, output_value_loss_history, 
                dist_metric_history, value_metric_history, victory_0, victory_1, victory_2))
        #
        #    
        # Model evaluation
        #
        #
        with open(file_name, 'a') as f:
            print('MODEL EVALUATION - Network vs. Old Network', file=f)

        # The current network faces the previous one.
        # If it does not win victory_rate % it completely
        # discards the current network.

        victory_0_eval_net = 0 # Draw
        victory_1_eval_net = 0 # Current model
        victory_2_eval_net = 0 # Old model

        for i in range(current_model.n_games_evaluate):
            # If "i" is even, current_model is Player 1, otherwise current_model is Player 2.
            # This helps reduce possible victory biases from player placements.
            if i % 2 == 0:
                _, who_won = self.play_single_game(current_model, old_model)
                if who_won == 1:
                    victory_1_eval_net += 1
                elif who_won == 2:
                    victory_2_eval_net += 1
                else:
                    victory_0_eval_net += 1
            else:
                _, who_won = self.play_single_game(old_model, current_model)
                if who_won == 2:
                    victory_1_eval_net += 1
                elif who_won == 1:
                    victory_2_eval_net += 1
                else:
                    victory_0_eval_net += 1

        data_net_vs_net_eval.append((victory_0_eval_net, victory_1_eval_net, victory_2_eval_net))

        necessary_won_games = (current_model.victory_rate * current_model.n_games_evaluate) / 100

        if victory_1_eval_net <= necessary_won_games:
            with open(file_name, 'a') as f:
                print('New model is worse...', file=f)
                print('New model victories: ', victory_1_eval_net, file=f)
                print('Old model victories: ', victory_2_eval_net, file=f)
                print('Draws: ', victory_0_eval_net, file=f)
        else:
            with open(file_name, 'a') as f:
                print('New model is better!', file=f)
                print('New model victories: ', victory_1_eval_net, file=f)
                print('Old model victories: ', victory_2_eval_net, file=f)
                print('Draws: ', victory_0_eval_net, file=f)

            # The new model is better. Therefore, evaluate it against
            # a list of vanilla UCTs and store the data for later analysis.

            # List of victories of each vanilla UCTs
            victory_0_eval = [0 for i in range(len(UCTs_eval))]
            victory_1_eval = [0 for i in range(len(UCTs_eval))]
            victory_2_eval = [0 for i in range(len(UCTs_eval))]

            for ucts in range(len(UCTs_eval)):
                with open(file_name, 'a') as f:
                    print('MODEL EVALUATION - Network vs. UCT - ', UCTs_eval[ucts].n_simulations,' simulations', file=f)

                for i in range(current_model.n_games_evaluate):
                    # If "i" is even, current_model is Player 1, otherwise current_model is Player 2.
                    # This helps reduce possible victory biases from player placements.
                    if i % 2 == 0:
                        _, who_won = self.play_single_game(current_model, UCTs_eval[ucts])
                        if who_won == 1:
                            victory_1_eval[ucts] += 1
                        elif who_won == 2:
                            victory_2_eval[ucts] += 1
                        else:
                            victory_0_eval[ucts] += 1
                    else:
                        _, who_won = self.play_single_game(UCTs_eval[ucts], current_model)
                        if who_won == 2:
                            victory_1_eval[ucts] += 1
                        elif who_won == 1:
                            victory_2_eval[ucts] += 1
                        else:
                            victory_0_eval[ucts] += 1

                with open(file_name, 'a') as f:
                    print('Net vs UCT - Network won', victory_1_eval[ucts],'time(s).', file=f)
                    print('Net vs UCT - UCT won', victory_2_eval[ucts],'time(s).', file=f)
                    print('Net vs UCT - Draws: ', victory_0_eval[ucts], file=f)

            list_of_n_simulations = [uct.n_simulations for uct in UCTs_eval]
            # Saving data
            data_net_vs_uct.append((victory_0_eval, victory_1_eval, victory_2_eval, list_of_n_simulations))

            elapsed_time = time.time() - start
            with open(file_name, 'a') as f:
                print('Time elapsed of this iteration: ', elapsed_time, file=f)

            stats = Statistic(data_net_vs_net_training, data_net_vs_net_eval, data_net_vs_uct, 
                    n_simulations = current_model.n_simulations, n_games = current_model.n_games, 
                    alphazero_iterations = current_model.alphazero_iterations, 
                     use_UCT_playout = use_UCT_playout, conv_number = conv_number)

            # New model is better, therefore we can copy current_model weights
            # to old_model to be used in the next AZ iteration.
            old_model.network.set_weights(current_model.network.get_weights()) 
            return stats, current_model, old_model

        # New model is worse, therefore we can copy old_model weights
        # to current_model to be used in the next AZ iteration.
        # This way, we are discarding everything current_model learned 
        #during the learning stage because it was worse than old_model.
        current_model.network.set_weights(old_model.network.get_weights()) 
        return [], current_model, old_model
               	
