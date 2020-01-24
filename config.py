class GameConfig:
    """ Configuration class for the Can't Stop game specs. """
    def __init__(self, n_players, dice_number, dice_value, column_range,
                    offset, initial_height):
        """
        - n_players is the number of players (At the moment, only 2 is possible).
        - dice_number is the number of dice used in the Can't Stop game.
        - dice_value is the number of faces of a single die.
        - column_range is a list denoting the range of the board game columns.
        - offset is the height difference between columns.
        - initial_height is the height of the columns at the border of the board.
        """
        self.n_players = n_players
        self.dice_number = dice_number
        self.dice_value = dice_value
        self.column_range = column_range 
        self.offset = offset
        self.initial_height = initial_height

class AlphaZeroConfig:
    """ Configuration class for the AlphaZero algorithm. """
    def __init__(self, c, n_simulations, n_games, n_games_evaluate, max_game_length,
                    victory_rate, alphazero_iterations, use_UCT_playout):
        """
        - c is the constant the balance exploration and exploitation.
        - n_simulations is the number of simulations made in the UCT algorithm.
        - n_games is the number of games played in the self-play scheme.
        - n_games_evaluate is the number of games played to evaluate the current
          network against the previous one.
        - max_game_length is the max number of plays in a game by both
          players (avoids infinite game).
        - victory_rate is the % of victories necessary for the new network to
          overwrite the previous one.
        - alphazero_iterations is the total number of iterations of the learning
          algorithm: selfplay -> training loop -> evaluate network (repeat).
        - use_UCT_playout: True will use standard UCT playout/simulation/rollout
          instead of getting the winning prediction from the network. False will
          use the information fetched from the network.
        """
        self.c = c
        self.n_simulations = n_simulations
        self.n_games = n_games
        self.n_games_evaluate = n_games_evaluate
        self.max_game_length = max_game_length
        self.victory_rate = victory_rate
        self.alphazero_iterations = alphazero_iterations
        self.use_UCT_playout = use_UCT_playout

class NetworkConfig:
    """ Configuration class for the neural network. """
    def __init__(self, reg, epochs, conv_number):
        """
        - reg is the L2 regularization parameter.
        - epochs is the number of training epochs.
        - conv_number is the number of convolutional layers in the network.
        """
        self.reg = reg
        self.epochs = epochs
        self.conv_number = conv_number