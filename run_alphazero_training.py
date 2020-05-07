from statistics import Statistic
from players.vanilla_uct_player import Vanilla_UCT
from players.net_uct_player import Network_UCT
from players.net_uct_with_playout_player import Network_UCT_With_Playout
from players.random_player import RandomPlayer
from experiment import Experiment
import sys
from models import define_model, define_model_experimental



def main():
    # Command line parameters: n_simulations, n_games, alpha_zero, 
    # conv_number, use_UCT_playout

    # Cluster configurations
    if int(sys.argv[1]) == 0: n_simulations = 2
    if int(sys.argv[1]) == 1: n_simulations = 100
    if int(sys.argv[1]) == 2: n_simulations = 250
    if int(sys.argv[1]) == 3: n_simulations = 500
    if int(sys.argv[2]) == 0: n_games = 10
    if int(sys.argv[2]) == 1: n_games = 100
    if int(sys.argv[2]) == 2: n_games = 250
    if int(sys.argv[2]) == 3: n_games = 500
    if int(sys.argv[3]) == 0: alphazero_iterations = 20
    if int(sys.argv[4]) == 0: conv_number = 1
    if int(sys.argv[4]) == 1: conv_number = 2
    if int(sys.argv[5]) == 0: use_UCT_playout = True
    if int(sys.argv[5]) == 1: use_UCT_playout = False

    #Config parameters

    c = 1
    epochs = 1
    reg = 0.01
    n_games_evaluate = 10
    victory_rate = 55
    mini_batch = 2048
    n_training_loop = 10
    dataset_size = 200
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

    file_name = str(n_simulations) + '_' + str(n_games) \
                + '_' + str(alphazero_iterations) + '_' + str(conv_number) \
                + '_' + str(use_UCT_playout) + '.txt'

    with open(file_name, 'a') as f:
        print('The arguments are: ' , str(sys.argv), file=f)

    experiment.play_alphazero(
                    player1, 
                    player2,  
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