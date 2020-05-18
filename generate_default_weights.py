import sys
import pickle
from models import define_model

def main():
    # Command line parameters: n_simulations, n_games, alpha_zero, 
    # conv_number, use_UCT_playout

    # Cluster configurations
    if int(sys.argv[1]) == 0: n_simulations = 10
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

    #Config parameters to create a model

    reg = 0.01
    
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
    '''
    
    model = define_model(
                        reg = reg, 
                        conv_number = conv_number, 
                        column_range = column_range, 
                        offset = offset, 
                        initial_height = initial_height, 
                        dice_value = dice_value
                        )

    current_weights_file = str(n_simulations) + '_' \
                + str(n_games) + '_' + str(alphazero_iterations) + '_' \
                + str(conv_number) + '_' + str(use_UCT_playout) \
                + '_currentweights'

    with open(current_weights_file, 'wb') as file:
        pickle.dump(model.get_weights(), file)

if __name__ == "__main__":
    main()