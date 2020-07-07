from players.player import Player
import numpy as np

class Rule_of_28_Player(Player):
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

