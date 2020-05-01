from statistics import Statistic
from players.vanilla_uct_player import Vanilla_UCT
from players.net_uct_player import Network_UCT
from players.net_uct_with_playout_player import Network_UCT_With_Playout
from players.random_player import RandomPlayer
from experiment import Experiment
import tensorflow as tf
import sys
import tkinter.filedialog
from os import listdir
from os.path import isfile, join
import re
import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import regularizers
from keras.losses import categorical_crossentropy

def define_model_experimental(reg, conv_number, column_range, offset, 
                                initial_height, dice_value):
    """Neural Network model implementation using Keras + Tensorflow."""

    # Calculating the channel dimensions given the board dynamics
    height = column_range[1] - column_range[0] + 1
    longest_column = (column_range[1] // 2) + 1
    width = initial_height + offset * (longest_column - column_range[0])
    # Calculating total number of actions possible (does not take into
    # consideration duplicate actions. Ex.: (2,3) and (3,2))
    temp = len(list(range(2, dice_value * 2 + 1)))
    n_actions = (temp*(1+temp))//2 + temp + 2

    n_channels = 6
    
    state_channels = Input(
                            shape = (n_channels, height, width), 
                            name='States_Channels_Input'
                            )
    valid_actions_dist = Input(
                                shape = (n_actions,), 
                                name='Valid_Actions_Input'
                                )

    zeropadding = keras.layers.ZeroPadding2D((2, 2))(state_channels)
    conv = Conv2D(
                    96, 
                    (4, 4), 
                    padding = "valid", 
                    kernel_initializer = 'glorot_normal', 
                    kernel_regularizer = regularizers.l2(reg), 
                    activation = 'relu', 
                    name = 'Conv_Layer'
                    )(zeropadding)
    conv2 = Conv2D(
                    96, 
                    (2, 2), 
                    padding = "valid", 
                    kernel_initializer = 'glorot_normal',
                    kernel_regularizer = regularizers.l2(reg), 
                    activation = 'relu', 
                    name = 'Conv_Layer2'
                    )(conv)
    batch1 = keras.layers.BatchNormalization()(conv2)
    act1 = keras.layers.Activation("relu")(batch1)
    flat = Flatten(name='Flatten_Layer')(act1)

    # Merge of the flattened channels (after pooling) and the valid action
    # distribution. Used only as input in the probability distribution head.
    merge = concatenate([flat, valid_actions_dist])

    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(
                                100, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'relu', 
                                name = 'FC_Prob_1'
                                )(merge)
    hidden_fc_prob_dist_2 = Dense(
                                100, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'relu', 
                                name = 'FC_Prob_2'
                                )(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(
                        n_actions, 
                        kernel_initializer = 'glorot_normal', 
                        kernel_regularizer = regularizers.l2(reg), 
                        activation = 'softmax', 
                        name = 'Output_Dist'
                        )(hidden_fc_prob_dist_2)
    
    #Value of a state
    hidden_fc_value_1 = Dense(
                        100, 
                        kernel_initializer = 'glorot_normal', 
                        kernel_regularizer = regularizers.l2(reg), 
                        activation = 'relu', 
                        name = 'FC_Value_1'
                        )(flat)
    output_value = Dense(
                        1, 
                        kernel_initializer = 'glorot_normal', 
                        kernel_regularizer = regularizers.l2(reg), 
                        activation = 'tanh', 
                        name = 'Output_Value'
                        )(hidden_fc_value_1)

    model = Model(
                    inputs=[state_channels, valid_actions_dist], 
                    outputs=[output_prob_dist, output_value]
                    )

    model.compile(
                loss=['categorical_crossentropy','mean_squared_error'],
                optimizer='adam', 
                metrics={'Output_Dist':'categorical_crossentropy', 
                            'Output_Value':'mean_squared_error'},
                loss_weights = [1, 1])
    return model 

def define_model(reg, conv_number, column_range, offset, 
                    initial_height, dice_value):
    """Neural Network model implementation using Keras + Tensorflow."""

    # Calculating the channel dimensions given the board dynamics
    height = column_range[1] - column_range[0] + 1
    longest_column = (column_range[1] // 2) + 1
    width = initial_height + offset * (longest_column - column_range[0])
    # Calculating total number of actions possible (does not take into
    # consideration duplicate actions. Ex.: (2,3) and (3,2))
    temp = len(list(range(2, dice_value * 2 + 1)))
    n_actions = (temp*(1+temp))//2 + temp + 2

    n_channels = 6

    state_channels = Input(
                        shape = (n_channels, height, width), 
                        name = 'States_Channels_Input'
                        )
    valid_actions_dist = Input(
                            shape = (n_actions,), 
                            name='Valid_Actions_Input'
                            )

    conv = Conv2D(
                filters = 10, 
                kernel_size = 2, 
                kernel_initializer = 'glorot_normal', 
                kernel_regularizer = regularizers.l2(reg), 
                activation = 'relu', 
                name = 'Conv_Layer'
                )(state_channels)
    if conv_number == 2:
        conv2 = Conv2D(
                    filters = 10, 
                    kernel_size = 2, 
                    kernel_initializer = 'glorot_normal',
                    kernel_regularizer = regularizers.l2(reg), 
                    activation = 'relu', 
                    name = 'Conv_Layer2'
                    )(conv)
    if conv_number == 1:
        flat = Flatten(name='Flatten_Layer')(conv)
    else:
        flat = Flatten(name='Flatten_Layer')(conv2)

    # Merge of the flattened channels and the valid action
    # distribution. Used only as input in the probability distribution head.
    merge = concatenate([flat, valid_actions_dist])

    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(
                                100, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'relu', 
                                name = 'FC_Prob_1'
                                )(merge)
    hidden_fc_prob_dist_2 = Dense(
                                100, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'relu', 
                                name = 'FC_Prob_2'
                                )(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(
                                n_actions, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'softmax', 
                                name = 'Output_Dist'
                                )(hidden_fc_prob_dist_2)
    
    #Value of a state
    hidden_fc_value_1 = Dense(
                            100, 
                            kernel_initializer = 'glorot_normal', 
                            kernel_regularizer = regularizers.l2(reg), 
                            activation = 'relu', 
                            name = 'FC_Value_1'
                            )(flat)
    hidden_fc_value_2 = Dense(
                            100, 
                            kernel_initializer = 'glorot_normal', 
                            kernel_regularizer = regularizers.l2(reg), 
                            activation = 'relu', 
                            name = 'FC_Value_2'
                            )(hidden_fc_value_1)
    output_value = Dense(
                        1, 
                        kernel_initializer = 'glorot_normal', 
                        kernel_regularizer = regularizers.l2(reg), 
                        activation = 'tanh', 
                        name = 'Output_Value'
                        )(hidden_fc_value_2)

    model = Model(
                inputs=[state_channels, valid_actions_dist], 
                outputs=[output_prob_dist, output_value]
                )

    model.compile(
                loss=['categorical_crossentropy','mean_squared_error'],
                optimizer='adam', 
                metrics={'Output_Dist':'categorical_crossentropy', 
                            'Output_Value':'mean_squared_error'},
                loss_weights = [1, 1])

    return model  

def main():
    # Command line parameters: n_simulations, n_games, alpha_zero, 
    # conv_number, use_UCT_playout

    # If the user does not pass any extra command line arguments,
    # then it will open the dialog to generate a report.
    if len(sys.argv) == 1:
        root = tkinter.Tk()
        root.withdraw()
        file_path = tkinter.filedialog.askdirectory()
        print(file_path)
        files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        valid_files = []
        for file in files:
            if 'h5' not in file:
                valid_files.append(file)

        # Sort the files using natural sorting
        # Source: https://stackoverflow.com/questions/5967500
        #            /how-to-correctly-sort-a-string-with-a-number-inside

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]

        valid_files.sort(key=natural_keys)

        data_net_vs_net_training = [] 
        data_net_vs_net_eval = [] 
        data_net_vs_uct = []
        n_simulations = None
        n_games = None
        alphazero_iterations = None
        use_UCT_playout = None
        conv_number = None
        for file in valid_files:
            file = file_path + '/' + file
            stats = Statistic()
            stats.load_from_file(file)
            data_net_vs_net_training.append(stats.data_net_vs_net_training[0]) 
            data_net_vs_net_eval.append(stats.data_net_vs_net_eval[0])
            # In case the new network was worse than the old one, the UCT data
            # does not exist, therefore there's no data
            if stats.data_net_vs_uct:
                data_net_vs_uct.append(stats.data_net_vs_uct[0]) 
            n_simulations = stats.n_simulations
            n_games = stats.n_games
            alphazero_iterations = stats.alphazero_iterations
            use_UCT_playout = stats.use_UCT_playout
            conv_number = stats.conv_number
        stats = Statistic(
                data_net_vs_net_training, data_net_vs_net_eval, 
                data_net_vs_uct, n_simulations, n_games,
                alphazero_iterations, use_UCT_playout, conv_number
                )
        stats.generate_report()
        exit()

    # Cluster configurations
    if int(sys.argv[1]) == 0: n_simulations = 2
    if int(sys.argv[1]) == 1: n_simulations = 100
    if int(sys.argv[1]) == 2: n_simulations = 250
    if int(sys.argv[1]) == 3: n_simulations = 500
    if int(sys.argv[2]) == 0: n_games = 1000
    if int(sys.argv[2]) == 1: n_games = 100
    if int(sys.argv[2]) == 2: n_games = 250
    if int(sys.argv[2]) == 3: n_games = 500
    if int(sys.argv[3]) == 0: alphazero_iterations = 150
    if int(sys.argv[4]) == 0: conv_number = 1
    if int(sys.argv[4]) == 1: conv_number = 2
    if int(sys.argv[5]) == 0: use_UCT_playout = True
    if int(sys.argv[5]) == 1: use_UCT_playout = False

    #Config parameters

    c = 1
    epochs = 1
    reg = 0.01
    n_games_evaluate = 100
    victory_rate = 55
    mini_batch = 2048
    n_training_loop = 100
    dataset_size = 1000
    '''
    # Toy version
    column_range = [2,6]
    offset = 2
    initial_height = 1
    n_players = 2
    dice_number = 4
    dice_value = 3
    max_game_length = 50
    '''
    # Original version
    column_range = [2,12]
    offset = 2
    initial_height = 2 
    n_players = 2
    dice_number = 4
    dice_value = 6 
    max_game_length = 500
    
    

    #Neural network specification
    current_model = define_model(
                                reg = reg, 
                                conv_number = conv_number, 
                                column_range = column_range, 
                                offset = offset, 
                                initial_height = initial_height, 
                                dice_value = dice_value
                                )
    old_model = define_model(
                            reg = reg, 
                            conv_number = conv_number, 
                            column_range = column_range, 
                            offset = offset, 
                            initial_height = initial_height, 
                            dice_value = dice_value
                            )
    old_model.set_weights(current_model.get_weights())

    if use_UCT_playout:
        player1 = Network_UCT_With_Playout(
                                    c = c, 
                                    n_simulations = n_simulations,  
                                    column_range = column_range, 
                                    offset = offset, 
                                    initial_height = initial_height, 
                                    dice_value = dice_value,
                                    network = current_model
                                    )
        player2 = Network_UCT_With_Playout(
                                    c = c, 
                                    n_simulations = n_simulations,  
                                    column_range = column_range, 
                                    offset = offset, 
                                    initial_height = initial_height, 
                                    dice_value = dice_value,
                                    network = old_model
                                    )
    else:
        player1 = Network_UCT(
                            c = c, 
                            n_simulations = n_simulations, 
                            column_range = column_range,
                            offset = offset, 
                            initial_height = initial_height, 
                            dice_value = dice_value, 
                            network = current_model
                            )
        player2 = Network_UCT(
                            c = c, 
                            n_simulations = n_simulations, 
                            column_range = column_range,
                            offset = offset, 
                            initial_height = initial_height, 
                            dice_value = dice_value, 
                            network = old_model
                            )

    experiment = Experiment(
                        n_players = n_players, 
                        dice_number = dice_number, 
                        dice_value = dice_value, 
                        column_range = column_range, 
                        offset = offset, 
                        initial_height = initial_height, 
                        max_game_length = max_game_length
                        )

    uct_evaluation_1 = Vanilla_UCT(
                                c = c, 
                                n_simulations = round(0.25 * n_simulations)
                                )
    uct_evaluation_2 = Vanilla_UCT(
                                c = c, 
                                n_simulations = round(0.5 * n_simulations)
                                )
    uct_evaluation_3 = Vanilla_UCT(
                                c = c, 
                                n_simulations =  n_simulations
                                )
    UCTs_eval = [uct_evaluation_1, uct_evaluation_2, uct_evaluation_3]

    file_name = str(n_simulations) + '_' + str(n_games) \
                + '_' + str(alphazero_iterations) + '_' + str(conv_number) \
                + '_' + str(use_UCT_playout) + '.txt'

    with open(file_name, 'a') as f:
        print('The arguments are: ' , str(sys.argv), file=f)

    experiment.play_alphazero(
                    player1, 
                    player2, 
                    UCTs_eval,  
                    use_UCT_playout = use_UCT_playout, 
                    reg = reg,
                    epochs = epochs, 
                    conv_number = conv_number,
                    alphazero_iterations = alphazero_iterations,
                    mini_batch = mini_batch,
                    n_training_loop = n_training_loop, 
                    n_games = n_games, 
                    n_games_evaluate = n_games_evaluate, 
                    victory_rate = victory_rate,
                    dataset_size = dataset_size)

if __name__ == "__main__":
    main()