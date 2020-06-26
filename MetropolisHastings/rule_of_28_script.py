import sys
sys.path.insert(0,'..')
from players.player import Player
import numpy as np 
import pickle
from players.vanilla_uct_player import Vanilla_UCT
from players.random_player import RandomPlayer
from players.uct_player import UCTPlayer
from game import Game

class Rule_of_28_Script(Player):
    """ 
    Heuristic proposed by the article 'A Generalized Heuristic for 
    Can’t Stop'.
    """

    def get_action(self, state):
        actions = state.available_moves()
        
        if actions == ['y', 'n']:
            score = 0
            neutral = state.neutral_positions
            neutral = [n[0] for n in neutral]

            for column in range(state.column_range[0], state.column_range[1]+1):
                if column not in neutral:
                    continue
                else:
                    advance = self.number_cells_advanced_this_round_for_col(state, column)
                    score += (advance + 1) * (abs(7 - column) + 1)

            if len(neutral) == 3:
                # If all neutral markers are in even columns
                if all([x % 2 == 0 for x in neutral]):
                    score -= 2
                # If all neutral markers are in odd columns
                elif all([x % 2 != 0 for x in neutral]):
                    score += 2

            if score >= 28:
                return 'n'
            else:
                return 'y'
        else:
            scores = np.zeros(len(actions))
            for i in range(len(scores)):
                scores[i] = self.calculate_action_score(actions[i], state)

            return actions[np.argmax(scores)]

    def calculate_action_score(self, action, state):

        neutral_positions = state.neutral_positions
        score = 0
        # Special case: double action (e.g.: (6,6))
        if len(action) == 2 and action[0] == action[1]:
            advance = 2
            is_new_neutral = True
            for neutral in neutral_positions:
                if neutral[0] == action:
                    is_new_neutral = False
            if is_new_neutral:
                markers = 1
            else:
                markers = 0
            score = advance * (6 - abs(7 - action[0])) - 6 * markers
            return score
        else:
            for a in action:
                advance = 1
                is_new_neutral = True
                for neutral in neutral_positions:
                    if neutral[0] == action:
                        is_new_neutral = False
                if is_new_neutral:
                    markers = 1
                else:
                    markers = 0
                score += advance * (6 - abs(7 - a)) - 6 * markers

            return score

    def number_cells_advanced_this_round_for_col(self, state, column):
        """
        Return the number of positions advanced in this round for a given
        column by the player.
        """
        counter = 0
        previously_conquered = -1
        neutral_position = -1
        list_of_cells = state.board_game.board[column]

        for i in range(len(list_of_cells)):
            if state.player_turn in list_of_cells[i].markers:
                previously_conquered = i
            if 0 in list_of_cells[i].markers:
                neutral_position = i
        if previously_conquered == -1 and neutral_position != -1:
            counter += neutral_position + 1
            for won_column in state.player_won_column:
                if won_column[0] == column:
                    counter += 1
        elif previously_conquered != -1 and neutral_position != -1:
            counter += neutral_position - previously_conquered
            for won_column in state.player_won_column:
                if won_column[0] == column:
                    counter += 1
        elif previously_conquered != -1 and neutral_position == -1:
            for won_column in state.player_won_column:
                if won_column[0] == column:
                    counter += len(list_of_cells) - previously_conquered
        return counter


def play_single_game(player1, player2, game, max_game_length):
    """
    Play a single game between player1 and player2.
    Return 1 if player1 won, 2 if player2 won or if the game reached the max
    number of iterations, it returns 0 representing a draw.
    """

    is_over = False
    rounds = 0
    # actions_taken actions in a row from a player. Used in UCTPlayer players. 
    # List of tuples (action taken, player turn, Game instance).
    # If players change turn, empty the list.
    actions_taken = []
    actions_from_player = 1

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
            if game.player_turn == 1 and isinstance(player1, UCTPlayer):
                if actions_from_player == game.player_turn:
                    chosen_play = player1.get_action(game, [])
                else:
                    chosen_play = player1.get_action(game, actions_taken)
            elif game.player_turn == 1 and not isinstance(player1, UCTPlayer):
                    chosen_play = player1.get_action(game)
            elif game.player_turn == 2 and isinstance(player2, UCTPlayer):
                if actions_from_player == game.player_turn:
                    chosen_play = player2.get_action(game, [])
                else:
                    chosen_play = player2.get_action(game, actions_taken)
            elif game.player_turn == 2 and not isinstance(player2, UCTPlayer):
                    chosen_play = player2.get_action(game)

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
            who_won = 0
            is_over = True
        else:
            who_won, is_over = game.is_finished()

    return who_won

'''
my_script = Rule_of_28_Script()
data = []
# If there is, read from it.
with open('dataset', "rb") as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break

n_errors = 0
for i in range(len(data)):
    state = data[i][0]
    oracle_play = data[i][1]
    chosen_play = my_script.get_action(state)
    print('available_moves = ', state.available_moves(), 'chosen_play = ', chosen_play, 'oracle_play = ', oracle_play)
    if chosen_play != oracle_play:
        n_errors += 1

errors_rate = n_errors / len(data)

print("Number of errors = ", n_errors, "(", errors_rate, "%)")
'''



# Original version
column_range = [2,12]
offset = 2
initial_height = 2 
n_players = 2
dice_number = 4
dice_value = 6 
max_game_length = 500

my_script = Rule_of_28_Script()
p1 = 0
p2 = 0
for i in range(1000):
    game = Game(n_players, dice_number, dice_value, column_range, offset, 
            initial_height)
    uct_vanilla_1 = Vanilla_UCT(c = 1, n_simulations = 50)
    uct_vanilla_2 = Vanilla_UCT(c = 1, n_simulations = 100)
    random_player = RandomPlayer()

    who_won = play_single_game(my_script, random_player, game, max_game_length)
    if who_won == 1:
        p1+=1
    elif who_won == 2:
        p2+=1
    #print('Winner from game', i, '= ', who_won)

print('p1 = ', p1)
print('p2 = ', p2)