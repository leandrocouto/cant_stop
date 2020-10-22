from players.player import Player
import numpy as np

class Solitaire_Rule_of_28_Player(Player):
    """ 
    Heuristic proposed by the article 'A Generalized Heuristic for 
    Can’t Stop'.
    """

    def __init__(self):
        self.score = 0
        self.progress_value = [0, 0, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6]
        self.move_value = [0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
        # Difficulty score
        self.odds = 2
        self.evens = -2
        self.highs = 4
        self.lows = 4
        self.marker = 6
        self.threshold = 28

    def get_action(self, state):
        actions = state.available_moves()
        
        if actions == ['y', 'n']:
            available_columns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            for neutral in state.neutral_positions:
                available_columns.remove(neutral[0])
            for finished in state.finished_columns:
                if finished in available_columns:
                    available_columns.remove(finished)
            # Check if the player will win the game if they choose 'n'
            clone_state = state.clone()
            clone_state.play('n')
            won_columns = len(clone_state.finished_columns)
            #This means if the player stop playing now, they will win the game
            if won_columns == 3:
                return 'n'
            elif state.n_neutral_markers != 3 and len(available_columns) > 0:
                return 'y'
            else:
                # Difficulty score
                neutral = [n[0] for n in state.neutral_positions]
                # If all neutral markers are in odd columns
                if all([x % 2 != 0 for x in neutral]):
                    self.score += self.odds
                # If all neutral markers are in even columns
                if all([x % 2 == 0 for x in neutral]):
                    self.score += self.evens
                # If all neutral markers are is "low" columns
                if all([x <= 7 for x in neutral]):
                    self.score += self.lows
                # If all neutral markers are is "high" columns
                if all([x >= 7 for x in neutral]):
                    self.score += self.highs

                if self.score >= self.threshold:
                    # Reset to zero the score for next player's round
                    self.score = 0
                    return 'n'
                else:
                    return 'y'
        else:
            scores = np.zeros(len(actions))
            for i in range(len(scores)):
                scores[i] = self.calculate_action_score(actions[i], state)

            index_chosen_action = np.argmax(scores)
            chosen_action = actions[index_chosen_action]

            # Incrementally adds the score that will be used in the yes/no action
            neutral = [n[0] for n in state.neutral_positions]
            # Special case: doubled action (e.g.: (6,6))
            if len(chosen_action) == 2 and chosen_action[0] == chosen_action[1]:
                # There's already a neutral marker in that column
                if chosen_action in neutral:
                    self.score += 2 * self.progress_value[chosen_action[0]]
                else:
                    self.score += 3 * self.progress_value[chosen_action[0]]
            else:
                for column in chosen_action:

                    # There's already a neutral marker in that column
                    if column in neutral:
                        self.score += self.progress_value[column]
                    else:
                        self.score += 2 * self.progress_value[column]
            return chosen_action

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
            score = advance * (self.move_value[action[0]]) - self.marker * markers
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
                score += advance * (self.move_value[a]) - self.marker * markers

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
            if state.player_id in list_of_cells[i].markers:
                previously_conquered = i
            if 0 in list_of_cells[i].markers:
                neutral_position = i
        if previously_conquered == -1 and neutral_position != -1:
            counter += neutral_position + 1
            for won_column in state.player_won_column:
                if won_column == column:
                    counter += 1
        elif previously_conquered != -1 and neutral_position != -1:
            counter += neutral_position - previously_conquered
            for won_column in state.player_won_column:
                if won_column == column:
                    counter += 1
        elif previously_conquered != -1 and neutral_position == -1:
            for won_column in state.player_won_column:
                if won_column == column:
                    counter += len(list_of_cells) - previously_conquered
        return counter