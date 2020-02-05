from statistics import Statistic
from players.vanilla_uct_player import Vanilla_UCT
from players.net_uct_player import Network_UCT
from players.net_uct_with_playout_player import Network_UCT_With_Playout
from players.random_player import RandomPlayer
from experiment import Experiment
import tensorflow as tf
import sys
import tkinter.filedialog

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import regularizers
from keras.losses import categorical_crossentropy

def define_model(reg, conv_number):
    """Neural Network model implementation using Keras + Tensorflow."""
    state_channels = Input(shape = (5,5,6), name='States_Channels_Input')
    valid_actions_dist = Input(shape = (32,), name='Valid_Actions_Input')

    conv = Conv2D(filters=10, kernel_size=2, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(reg), activation='relu', name='Conv_Layer')(state_channels)
    if conv_number == 2:
        conv2 = Conv2D(filters=10, kernel_size=2, kernel_initializer='glorot_normal',kernel_regularizer=regularizers.l2(reg), activation='relu', name='Conv_Layer2')(conv)
    if conv_number == 1:
        flat = Flatten(name='Flatten_Layer')(conv)
    else:
        flat = Flatten(name='Flatten_Layer')(conv2)

    # Merge of the flattened channels (after pooling) and the valid action
    # distribution. Used only as input in the probability distribution head.
    merge = concatenate([flat, valid_actions_dist])

    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(100, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(reg), activation='relu', name='FC_Prob_1')(merge)
    hidden_fc_prob_dist_2 = Dense(100, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(reg), activation='relu', name='FC_Prob_2')(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(32, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(reg), activation='softmax', name='Output_Dist')(hidden_fc_prob_dist_2)
    
    #Value of a state
    hidden_fc_value_1 = Dense(100, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(reg), activation='relu', name='FC_Value_1')(flat)
    hidden_fc_value_2 = Dense(100, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(reg), activation='relu', name='FC_Value_2')(hidden_fc_value_1)
    output_value = Dense(1, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(reg), activation='tanh', name='Output_Value')(hidden_fc_value_2)

    model = Model(inputs=[state_channels, valid_actions_dist], outputs=[output_prob_dist, output_value])

    model.compile(loss=['categorical_crossentropy','mean_squared_error'], 
                        optimizer='adam', metrics={'Output_Dist':'categorical_crossentropy', 'Output_Value':'mean_squared_error'})
    return model  

def main():
    # Command line parameters: n_simulations, n_games, alpha_zero, conv_number, use_UCT_playout
    # If the user does not pass any extra command line arguments,
    # then it will open the dialog to generate graphs.
    if len(sys.argv) == 1:
        root = tkinter.Tk()
        root.withdraw()
        file_path = tkinter.filedialog.askopenfilename()
        print(file_path)
        stats = Statistic()
        stats.load_from_file(file_path)
        stats.generate_graphs()
        exit()

    # Cluster configurations
    if int(sys.argv[1]) == 0: n_simulations = 10
    if int(sys.argv[1]) == 1: n_simulations = 100
    if int(sys.argv[1]) == 2: n_simulations = 300
    if int(sys.argv[2]) == 0: n_games = 10
    if int(sys.argv[2]) == 1: n_games = 500
    if int(sys.argv[2]) == 2: n_games = 1000
    if int(sys.argv[3]) == 0: alphazero_iterations = 3
    if int(sys.argv[3]) == 1: alphazero_iterations = 100
    if int(sys.argv[4]) == 0: conv_number = 1
    if int(sys.argv[4]) == 1: conv_number = 2
    if int(sys.argv[5]) == 0: use_UCT_playout = True
    if int(sys.argv[5]) == 1: use_UCT_playout = False

    print('The arguments are: ' , str(sys.argv))

    #Neural network specification
    current_model = define_model(reg = 0.01, conv_number = conv_number)
    old_model = define_model(reg = 0.01, conv_number = conv_number)
    old_model.set_weights(current_model.get_weights())

    player1 = Network_UCT(c = 10, n_simulations = n_simulations, n_games = n_games, n_games_evaluate = 1,
                    victory_rate = 55, alphazero_iterations = alphazero_iterations, column_range = [2,6],
                    offset = 2, initial_height = 1, network = current_model)

    if use_UCT_playout:
        player2 = Network_UCT_With_Playout(c = 10, n_simulations = n_simulations, n_games = n_games, n_games_evaluate = 1,
                    victory_rate = 55, alphazero_iterations = alphazero_iterations, column_range = [2,6],
                    offset = 2, initial_height = 1, network = old_model)
    else:
        player2 = Network_UCT(c = 10, n_simulations = n_simulations, n_games = n_games, n_games_evaluate = 1,
                    victory_rate = 55, alphazero_iterations = alphazero_iterations, column_range = [2,6],
                    offset = 2, initial_height = 1, network = old_model)

    experiment = Experiment(n_players = 2, dice_number = 4, dice_value = 3, column_range = [2,6],
                    offset = 2, initial_height = 1, max_game_length = 50)

    player1 = Vanilla_UCT(c = 10, n_simulations = n_simulations)
    player2 = RandomPlayer()

    for _ in range(10):
        _, who_won = experiment.play_single_game(player1, player2)
        print('Who won: ', who_won)

    #for count in range(3):
    #    stats, player1, player2 = experiment.play_alphazero_iteration(player2, player1, use_UCT_playout = use_UCT_playout, 
    #                                                                    epochs = 1, conv_number = conv_number)  
    #    if stats != []:
    #        stats.save_to_file(count)
    #        stats.save_model_to_file(player1.network, count)

if __name__ == "__main__":
    main()