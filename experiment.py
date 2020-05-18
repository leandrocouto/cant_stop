from game import Game
import time
import copy
import os
import pickle
import psutil
import os.path
import gc
from players.vanilla_uct_player import Vanilla_UCT
from players.alphazero_player import AlphaZeroPlayer
from players.uct_player import UCTPlayer
from statistics import Statistic
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf

class Experiment:

    def __init__(self, n_players, dice_number, dice_value, column_range,
        offset, initial_height, max_game_length, reg, conv_number, n_cores):
        """n_cpus is the number of cores used for parallel computations. """
        self.n_players = n_players
        self.dice_number = dice_number
        self.dice_value = dice_value
        self.column_range = column_range 
        self.offset = offset
        self.initial_height = initial_height
        self.max_game_length = max_game_length
        self.reg = reg
        self.conv_number = conv_number
        self.n_cores = n_cores

    def _play_single_game(self, args):
        """
        Play a single game between player1 and player2.
        Return the data of the game in a NN channel format and who won.
        If the game reaches the maximum number of iterations, it returns 0
        0 representing a draw. 
        Return an empty list if none of the players are instances of 
        AlphaZeroPlayer or the data collected from the game otherwise.
        Return 1 if player1 won and 2 if player2 won.
        """

        player1 = args[0]
        player2 = args[1]
        type_of_game = args[2]
        player1_weights = args[3]
        player2_weights = args[4]
        # Selfplay
        if type_of_game == 's':
            from models import define_model
            current_model = define_model(
                                    reg = self.reg, 
                                    conv_number = self.conv_number, 
                                    column_range = self.column_range, 
                                    offset = self.offset, 
                                    initial_height = self.initial_height, 
                                    dice_value = self.dice_value
                                    )
            
            copy_model = define_model(
                                    reg = self.reg, 
                                    conv_number = self.conv_number, 
                                    column_range = self.column_range, 
                                    offset = self.offset, 
                                    initial_height = self.initial_height, 
                                    dice_value = self.dice_value
                                    )

            player1.network = current_model
            player2.network = copy_model
                
            player1.network.set_weights(player1_weights)
            player2.network.set_weights(player1_weights)
        # Evaluation vs network
        elif type_of_game == 'en':
            from models import define_model
            current_model = define_model(
                                    reg = self.reg, 
                                    conv_number = self.conv_number, 
                                    column_range = self.column_range, 
                                    offset = self.offset, 
                                    initial_height = self.initial_height, 
                                    dice_value = self.dice_value
                                    )
            
            old_model = define_model(
                                    reg = self.reg, 
                                    conv_number = self.conv_number, 
                                    column_range = self.column_range, 
                                    offset = self.offset, 
                                    initial_height = self.initial_height, 
                                    dice_value = self.dice_value
                                    )

            player1.network = current_model
            player2.network = old_model
            player1.network.set_weights(player1_weights)
            player2.network.set_weights(player2_weights)
        
        # Evaluation vs UCTs
        elif type_of_game == 'eu':
            from models import define_model
            network = define_model(
                                    reg = self.reg, 
                                    conv_number = self.conv_number, 
                                    column_range = self.column_range, 
                                    offset = self.offset, 
                                    initial_height = self.initial_height, 
                                    dice_value = self.dice_value
                                    )
            if player1_weights != None:
                player1.network = network
                player1.network.set_weights(player1_weights)
            else:
                player2.network = network
                player2.network.set_weights(player2_weights)

        data_of_a_game = []
        game = Game(self.n_players, self.dice_number, self.dice_value, 
                    self.column_range, self.offset, self.initial_height)

        is_over = False
        rounds = 0
        # actions_taken actions in a row from a player. 
        # List of tuples (action taken, player turn, Game instance).
        # If players change turn, empty the list.
        actions_taken = []
        actions_from_player = 1

        # Loop of the game
        while not is_over:
            rounds += 1
            # Collecting data for later input to the NN if any of the players 
            # are subclasses of AlphaZeroPlayer.
            if isinstance(player1, AlphaZeroPlayer) and game.player_turn == 1:
                channel_valid = player1.valid_positions_channel(
                                    self.column_range, self.offset, 
                                    self.initial_height
                                    )
                channel_finished_1, channel_finished_2 = \
                                player1.finished_columns_channels(
                                                    game, channel_valid
                                                    )
                channel_won_column_1, channel_won_column_2 = \
                                player1.player_won_column_channels(
                                                    game, channel_valid
                                                    )
                channel_turn = player1.player_turn_channel(game, channel_valid)
                list_of_channels = [channel_valid, 
                                    channel_finished_1, channel_finished_2,
                                    channel_won_column_1, channel_won_column_2,
                                    channel_turn
                                    ]
            elif isinstance(player2, AlphaZeroPlayer) and game.player_turn == 2:
                channel_valid = player2.valid_positions_channel(
                                    self.column_range, self.offset, 
                                    self.initial_height
                                    )
                channel_finished_1, channel_finished_2 = \
                                player2.finished_columns_channels(
                                                    game, channel_valid
                                                    )
                channel_won_column_1, channel_won_column_2 = \
                                player2.player_won_column_channels(
                                                    game, channel_valid
                                                    )
                channel_turn = player2.player_turn_channel(game, channel_valid)
                list_of_channels = [channel_valid, 
                                    channel_finished_1, channel_finished_2,
                                    channel_won_column_1, channel_won_column_2,
                                    channel_turn
                                    ]
            moves = game.available_moves()
            if game.is_player_busted(moves):
                actions_taken = []
                actions_from_player = game.player_turn
                continue
            else:
                if game.player_turn == 1:
                    if actions_from_player == game.player_turn:
                        chosen_play = player1.get_action(game, [])
                    else:
                        chosen_play = player1.get_action(game, actions_taken)
                    dist_probability = []
                    if isinstance(player1, UCTPlayer):
                        dist_probability = player1.get_dist_probability()
                else:
                    if actions_from_player == game.player_turn:
                        chosen_play = player2.get_action(game, [])
                    else:
                        chosen_play = player2.get_action(game, actions_taken)
                    dist_probability = []
                    if isinstance(player2, UCTPlayer):
                        dist_probability = player2.get_dist_probability()
                if isinstance(player1, AlphaZeroPlayer) \
                    and isinstance(player2, AlphaZeroPlayer):
                    # Collecting data for network
                    current_play = [list_of_channels, dist_probability]
                    data_of_a_game.append(current_play)

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
                actions_taken.append((chosen_play, actual_player, 
                                        game.clone())
                                        )

            # if the game has reached its max number of plays, end the game
            # and who_won receives 0, which means no players won.

            if rounds > self.max_game_length:
                who_won = 0
                is_over = True
            else:
                who_won, is_over = game.is_finished()

        # Game is over, so resets the trees
        player1.reset_tree()
        player2.reset_tree()
        return data_of_a_game, who_won

    def _selfplay(self, current_model, current_weights, dataset_for_network, 
        n_games, file_name):

        with open(file_name, 'a') as f:
            print('SELFPLAY', file=f)

        start_selfplay = time.time()
        # 0 -> draw
        victory_0 = 0
        victory_1 = 0
        victory_2 = 0

        copy_model = current_model.clone()

        # ProcessPoolExecutor() will take care of joining() and closing()
        # the processes after they are finished.
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Specify which arguments will be used for each parallel call
            args = (
                    (current_model, copy_model, 's', current_weights, None) 
                    for _ in range(n_games)
                    )
            # data is a list of 2-tuples = (data_of_a_game, who_won) 
            results = executor.map(self._play_single_game, args)
        data_of_all_games = []
        for result in results:
            data_of_all_games.append(result)
        
        for single_game in data_of_all_games:
            data_of_a_game = single_game[0]
            who_won = single_game[1]
            if who_won == 1:
                    victory_1 += 1
            elif who_won == 2:
                victory_2 += 1
            else:
                victory_0 += 1
            # Store who actually won in all states
            for state in data_of_a_game:
                if who_won == 1:
                    state.append(1)
                elif who_won == 2:
                    state.append(-1)
            # Save the data only if it was not a draw
            if who_won != 0:
                dataset_for_network.append(data_of_a_game)
        
        del copy_model

        elapsed_time_selfplay = time.time() - start_selfplay

        with open(file_name, 'a') as f:
            print('    Selfplay - Player 1 won', victory_1, 'time(s).', \
                    file=f)
            print('    Selfplay - Player 2 won', victory_2, 'time(s).', \
                    file=f)
            print('    Selfplay - Ties:', victory_0, file=f)
            print('    Time elapsed in Selfplay:', elapsed_time_selfplay,\
                    file=f)
            print('    Average time of a game: ', elapsed_time_selfplay \
                    / n_games, 's', sep = '', file=f)

        return victory_0, victory_1, victory_2

    def _training(self, args):
        current_model = args[0]
        current_weights = args[1]
        dataset_for_network = args[2]
        n_training_loop = args[3]
        mini_batch  = args[4]
        epochs = args[5]
        victory_0 = args[6]
        victory_1 = args[7]
        victory_2 = args[8]
        file_name  = args[9]
        with open(file_name, 'a') as f:
                print('TRAINING LOOP', file=f)
            
        start_training = time.time()


        # Transform the dataset collected into network input
        channels_input, valid_actions_dist_input, dist_probs_label, \
            who_won_label = current_model.transform_dataset_to_input(
                                            dataset_for_network
                                            )
        
        from models import define_model
        model = define_model(
                            reg = self.reg, 
                            conv_number = self.conv_number, 
                            column_range = self.column_range, 
                            offset = self.offset, 
                            initial_height = self.initial_height, 
                            dice_value = self.dice_value
                            )
        model.set_weights(current_weights)
        current_model.network = model
        
        for i in range(n_training_loop):
            # Sample random mini_batch inputs for training
            x_train, y_train = current_model.sample_input(
                                channels_input, valid_actions_dist_input, 
                                dist_probs_label, who_won_label, 
                                mini_batch
                                )

            history_callback = current_model.network.fit(
                                x_train, y_train, epochs = epochs, 
                                shuffle = True, verbose = 0
                                )

            if i == n_training_loop - 1:
                # Saving data
                loss = sum(history_callback.history["loss"]) \
                                / len(history_callback.history["loss"])
                dist_metric = \
                        sum(history_callback.history[
                            "Output_Dist_categorical_crossentropy"
                            ]) \
                            / len(history_callback.history[
                                "Output_Dist_categorical_crossentropy"
                                ])
                value_metric = \
                        sum(history_callback.history[
                            "Output_Value_mean_squared_error"
                            ]) \
                            / len(history_callback.history[
                                "Output_Value_mean_squared_error"
                                ])
        
        training_tuple = (
                        loss, dist_metric, value_metric,
                        victory_0, victory_1, victory_2
                        )

        elapsed_time_training = time.time() - start_training
        
        current_weights = current_model.network.get_weights()
        current_model.network = None


        with open(file_name, 'a') as f:
            print('    Time elapsed of training:',
                elapsed_time_training, file=f)
            
            print('    Total loss:', loss, file=f)
            print('    Dist. error:', dist_metric, file=f)
            print('    Value error:', value_metric, file=f)

        return current_weights, training_tuple

    def _evaluation(self, current_model, cur_weights, old_weights,
        data_net_vs_net_eval, n_games_evaluate, victory_rate, file_name):

        with open(file_name, 'a') as f:
                print('MODEL EVALUATION - Network vs. Old Network', file=f)

        # The current network faces the previous one.
        # If it does not win victory_rate % it completely
        # discards the current network.

        victory_0_eval = 0 # Draw
        victory_1_eval = 0 # Current model
        victory_2_eval = 0 # Old model

        start_eval = time.time()

        old_model = current_model.clone()
        # We do n_games//2 parallel games. Each of the operations we switch 
        # who is the first player to avoid first player winning bias.

        # ProcessPoolExecutor() will take care of joining() and closing()
        # the processes after they are finished.
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Specify which arguments will be used for each parallel call
            args = (
                    (current_model, old_model, 'en', cur_weights, old_weights) 
                    for _ in range(n_games_evaluate//2)
                    )
            results_1 = executor.map(self._play_single_game, args)

        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Specify which arguments will be used for each parallel call
            args = (
                    (old_model, current_model, 'en', old_weights, cur_weights) 
                    for _ in range(n_games_evaluate//2)
                    )   
            results_2 = executor.map(self._play_single_game, args)
        
        for result in results_1:
            if result[1] == 1:
                victory_1_eval += 1
            elif result[1] == 2:
                victory_2_eval += 1
            else:
                victory_0_eval += 1

        for result in results_2:
            if result[1] == 2:
                victory_1_eval += 1
            elif result[1] == 1:
                victory_2_eval += 1
            else:
                victory_0_eval += 1
        
        elapsed_time_eval = time.time() - start_eval

        data_net_vs_net_eval.append(
            (victory_0_eval, victory_1_eval, victory_2_eval)
            )

        necessary_won_games = (victory_rate * n_games_evaluate) / 100

        if victory_1_eval < necessary_won_games:
            with open(file_name, 'a') as f:
                print('    New model is worse...', file=f)
                print('    New model victories:', victory_1_eval, file=f)
                print('    Old model victories:', victory_2_eval, file=f)
                print('    Draws:', victory_0_eval, file=f)
                print('    Time elapsed in evaluation (Net vs. Net):', 
                        elapsed_time_eval, file=f)
                print('    Average time of a game: ', \
                        elapsed_time_eval / n_games_evaluate, \
                        's', sep = '', file=f)

            return False
        else:
            with open(file_name, 'a') as f:
                print('    New model is better!', file=f)
                print('    New model victories:', victory_1_eval, file=f)
                print('    Old model victories:', victory_2_eval, file=f)
                print('    Draws:', victory_0_eval, file=f)
                print('    Time elapsed in evaluation (Net vs. Net):', 
                        elapsed_time_eval, file=f)
                print('    Average time of a game: ', \
                        elapsed_time_eval / n_games_evaluate, \
                        's', sep = '', file=f)

            return True

    def play_alphazero(self, current_model, current_weights, use_UCT_playout, 
        epochs, alphazero_iterations, mini_batch, n_training_loop, n_games, 
        n_games_evaluate, victory_rate, dataset_size, iteration):
        """
        - current_model is an instance of AlphaZeroPlayer.
        - epochs is the number of epochs usued in the training stage.
        - alphazero_iterations is the total number of iterations of the 
          learning algorithm: selfplay -> training loop -> evaluate network.
        - mini_batch is the number of data sampled from the whole dataset for 
          one single training iteration.
        - n_training_loop is the number of training iterations after self-play.
        - n_games is the number of games played in the self-play stage.
        - n_games_evaluate is the number of games played in the evaluation
          stage.
        - victory_rate is the % of victories necessary for the new network to
          overwrite the previous one.
        - dataset_size is the max nubmer of games stored in memory for 
          training. 
        - iteration is an integer referring to which iteration AZ is at the
          moment.
        """

        

        file_name = str(current_model.n_simulations) + '_' + str(n_games) \
                + '_' + str(alphazero_iterations) + '_' \
                + str(self.conv_number) + '_' + str(use_UCT_playout) + '.txt'

        dataset_file = str(current_model.n_simulations) + '_' + str(n_games) \
                + '_' + str(alphazero_iterations) + '_' \
                + str(self.conv_number) + '_' + str(use_UCT_playout) \
                + '_dataset'

        current_weights_file = str(current_model.n_simulations) + '_' \
                + str(n_games) + '_' + str(alphazero_iterations) + '_' \
                + str(self.conv_number) + '_' + str(use_UCT_playout) \
                + '_currentweights'
        old_weights_file = str(current_model.n_simulations) + '_' \
                + str(n_games) + '_' + str(alphazero_iterations) + '_' \
                + str(self.conv_number) + '_' + str(use_UCT_playout) \
                + '_oldweights'
        

        with open(file_name, 'a') as f:
            print('ALPHAZERO ITERATION -', iteration, file=f)
        
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / 1000000
        with open(file_name, 'a') as f:
            print('Current usage of RAM (mb): ', memory, file=f)

        # dataset_for_network stores a list of info used as input for the 
        # network.
        # A list of: states of the game, distribution probability of the state
        # returned by the UCT and who won the game this state is in.
        dataset_for_network = []

        # If there is a file that contains data for training, read it
        if os.path.exists(dataset_file):
            with open(dataset_file, 'rb') as file:
                dataset_for_network = pickle.load(file)

        if current_weights == None:
            raise Exception('You must run generate_default_weights.py.' + \
                            ' This is needed for the first call of' + \
                            ' selfplay in order to all concurrent calls' + \
                            ' run with the same network weights.'
                            )

        # Stores data from net vs net in training for later analysis.
        data_net_vs_net_training = []
        # Stores data from net vs net in evaluation for later analysis.
        data_net_vs_net_eval = []

        start = time.time()

        #
        #
        # Self-play
        #
        #

        victory_0, victory_1, victory_2 = self._selfplay(
                                                        current_model,
                                                        current_weights, 
                                                        dataset_for_network, 
                                                        n_games,
                                                        file_name
                                                        )
        
        # This means all of the selfplay games (from the first iteration) 
        # ended in a draw.
        # This is not interesting since it does not add any valued info 
        # for the network training. Stops this iteration.
        if len(dataset_for_network) == 0:
            with open(file_name, 'a') as f:
                print('    All selfplay games ended in a draw.' 
                        + ' Stopping current iteration.', file=f)
            return

        #
        #
        # Training
        #
        #

        

        #old_weights = None
        #old_model = current_model.clone()

        # Save the model weights before training for later evaluation
        with open(old_weights_file, 'wb') as file:
            pickle.dump(current_weights, file)

        current_dataset_size =  len(dataset_for_network)
        to_delete = current_dataset_size - dataset_size

        # If the current dataset is bigger than dataset_size, then removes
        # the oldest games accordingly.
        if current_dataset_size > dataset_size:
            del dataset_for_network[:to_delete]

        # This is a little hack in order to tensorflow to not be stuck in a
        # deadlock at net evaluation. Apparently, for parallel calls to work
        # with tf.keras models, no tf.keras models should be instantiated in
        # the main process (such as the way we did in the selfplay stage, we
        # instantiate the models inside each parallel process). Therefore, 
        # we call the training stage in a single "parallel" call; this way,
        # the define_model() will be called in a separate process rather than
        # the main one so we are able to run the net evaluation later with no
        # deadlocks.
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Specify which arguments will be used for each parallel call
            args = (
                    (current_model,
                    current_weights,
                    dataset_for_network,
                    n_training_loop, 
                    mini_batch,
                    epochs,
                    victory_0, 
                    victory_1, 
                    victory_2, 
                    file_name) 
                    for _ in range(1)
                    )
            # 2-tuple = (current_weights, training_tuple) 
            results = executor.map(self._training, args)

        for result in results:
            current_weights = result[0]
            training_tuple = result[1]

        data_net_vs_net_training.append(training_tuple)

        # Save the current model weights after training for later evaluation.
        # This seems redundant since we read from file right after, but it is
        # stored in file in case of unexpected problems for logging purposes.
        with open(current_weights_file, 'wb') as file:
            pickle.dump(current_weights, file)

        # Since the selfplay data is not important from now on, free the 
        # memory and save the data in a file for later usage in the next
        # AZ iteration.
        with open(dataset_file, 'wb') as file:
            pickle.dump(dataset_for_network, file)
        dataset_for_network = []

        # Load from file current and old weights for evaluation
        if os.path.exists(current_weights_file):
            with open(current_weights_file, 'rb') as file:
                current_weights = pickle.load(file)
        else:
            raise Exception('Current weights file was not located.')

        if os.path.exists(old_weights_file):
            with open(old_weights_file, 'rb') as file:
                old_weights = pickle.load(file)
        else:
            raise Exception('Old weights file was not located.')

        old_model = current_model.clone()

        #
        #    
        # Model evaluation
        #
        #

        won = self._evaluation(
                            current_model,
                            current_weights,
                            old_weights,
                            data_net_vs_net_eval, 
                            n_games_evaluate, 
                            victory_rate,
                            file_name
                            )

        if not won:
            # New model is worse, therefore we can copy old_model weights
            # to current_model to be used in the next AZ iteration.
            # This way, we are discarding everything current_model learned 
            #during the learning stage because it was worse than old_model.
            current_weights = old_weights
            # Save the current model weights with the old weights since it has
            # lost against the old weights.
            with open(current_weights_file, 'wb') as file:
                pickle.dump(current_weights, file)

        # Saves this iteration's data to file
        stats = Statistic(
                data_net_vs_net_training, 
                data_net_vs_net_eval, 
                None, 
                n_simulations = current_model.n_simulations, 
                n_games = n_games, 
                alphazero_iterations = alphazero_iterations, 
                use_UCT_playout = use_UCT_playout, 
                conv_number = self.conv_number
                )
        stats.save_to_file(iteration, won = won)
        stats.save_model_to_file(current_weights, iteration, won = won)
        stats.save_player_config(current_model, iteration, won = won)

        # At this point, we don't need the local files that stores current
        # and old files anmore since the weights are properly stored in
        # another folder, so delete them.
        os.remove(current_weights_file)
        os.remove(old_weights_file)

        elapsed_time = time.time() - start
        import gc
        gc.collect()

        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / 1000000
        with open(file_name, 'a') as f:
            print('Time elapsed of this AZ iteration: ', 
                elapsed_time, file=f)
            print('Current usage of RAM (mb): ', memory, file=f)
            print(file=f)
        

    def play_network_versus_UCT(self, network, weights, stat, networks_dir, 
        prefix_name, UCTs_eval, n_games_evaluate):
        """
        Play the network against three baseline UCTs and save all the data
        accordingly.
        
        - network is an AlphaZeroPlayer that won against a previous network 
          and for evaluation purposes will now face the UCTs.
        - stat is an instance of Statistics that stores data related to this
          player. It is used to generate a report.
        - networks_dir is the directory where all info about the current 
          experiment is located.
        - prefix_name is a string used to identify the configs used in this 
          experiment, as well as the AZ iteration it corresponds to.
        - UCTs_eval is a list of Vanilla_UCT players used in evaluation.
        - n_games_evaluate is the number of games played between the network
          and the vanilla UCTs.
        """

        start = time.time()

        file_name = networks_dir + '/results_uct/'
        file_name_data = networks_dir + '/results_uct/' + prefix_name + '_data'

        # Create the target directory if it is not created yet
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        file_name = file_name + prefix_name + '_log_uct.txt'
        # Stores data from net vs uct in evaluation for later analysis.
        data_net_vs_uct = []

        # List of victories of each vanilla UCTs
        victory_0_eval = [0 for i in range(len(UCTs_eval))]
        victory_1_eval = [0 for i in range(len(UCTs_eval))]
        victory_2_eval = [0 for i in range(len(UCTs_eval))]

        iteration = prefix_name.rsplit('_', 1)[-1]

        with open(file_name, 'a') as f:
                print('ALPHAZERO ITERATION -', iteration, file=f)

        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / 1000000
        with open(file_name, 'a') as f:
            print('Current usage of RAM (mb): ', memory, file=f)

        for ucts in range(len(UCTs_eval)):
            with open(file_name, 'a') as f:
                print('MODEL EVALUATION - Network vs. UCT - ', 
                    UCTs_eval[ucts].n_simulations,' simulations', file=f)

            start_evaluate_uct = time.time()
            # We do n_games//2 parallel games. Each of the operations we switch 
            # who is the first player to avoid first player winning bias.
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Specify which arguments will be used for each parallel call
                args = (
                        (network, UCTs_eval[ucts], 'eu', weights, None) 
                        for _ in range(n_games_evaluate//2)
                        )
                results_1 = executor.map(self._play_single_game, args)

            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                # Specify which arguments will be used for each parallel call
                args = (
                        (UCTs_eval[ucts], network, 'eu', None, weights) 
                        for _ in range(n_games_evaluate//2)
                        )   
                results_2 = executor.map(self._play_single_game, args)

            for result in results_1:
                if result[1] == 1:
                    victory_1_eval[ucts] += 1
                elif result[1] == 2:
                    victory_2_eval[ucts] += 1
                else:
                    victory_0_eval[ucts] += 1

            for result in results_2:
                if result[1] == 2:
                    victory_1_eval[ucts] += 1
                elif result[1] == 1:
                    victory_2_eval[ucts] += 1
                else:
                    victory_0_eval[ucts] += 1

            elapsed_time_evaluate_uct = time.time() - start_evaluate_uct

            with open(file_name, 'a') as f:
                print('    Net vs UCT - Network won', 
                    victory_1_eval[ucts],'time(s).', file=f)
                print('    Net vs UCT - UCT won', 
                    victory_2_eval[ucts],'time(s).', file=f)
                print('    Net vs UCT - Draws:', 
                    victory_0_eval[ucts], file=f)
                print('    Time elapsed in evaluation (Net vs. UCT):', 
                    elapsed_time_evaluate_uct, file=f)
                print('    Average time of a game: ', \
                    elapsed_time_evaluate_uct / n_games_evaluate, \
                    's', sep = '', file=f)

        gc.collect()
        elapsed_time = time.time() - start

        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / 1000000
        with open(file_name, 'a') as f:
            print('Time elapsed of this evaluation: ', elapsed_time, file=f)
            print('Current usage of RAM (mb): ', memory, file=f)
            print(file=f)

        list_of_n_simulations = [uct.n_simulations for uct in UCTs_eval]
        # Saving data
        data_net_vs_uct.append(
            (victory_0_eval, victory_1_eval, victory_2_eval, 
            list_of_n_simulations)
            )
        stat.data_net_vs_uct = data_net_vs_uct
        # Save the data of uct
        with open(file_name_data, 'wb') as file:
                pickle.dump(data_net_vs_uct, file)
        # Save the updated stats file
        with open(networks_dir + '/results_uct/' + prefix_name, 'wb') as file:
                pickle.dump(stat.__dict__, file)
