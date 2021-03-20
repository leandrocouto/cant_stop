from players.player import Player
import numpy as np
import random

class DSLGlennPlayer(Player):
    def get_action(self, state):
         import numpy as np
         actions = state.available_moves()
         a = 0
         b = 0
         c = 0
         scores = np.zeros(len(actions))
         if 'y' in actions:
             if self.will_player_win_after_n(state):
                 return 'n'
             elif self.are_there_available_columns_to_play(state):
                 return 'y'
             else:
                 score = self.calculate_score(state, [7, 7, 3, 2, 2, 1]) + self.calculate_difficulty_score(state, 7, 1, 6, 5)
                 if score >= 29:
                     return 'n'
                 else:
                     return 'y'
         else:
             for i in range(len(scores)):
                scores[i] = self.advance(actions[i]) * self.get_cols_weights(actions[i]) - 6 * self.is_new_neutral(state, actions[i])
             return actions[np.argmax(scores)]

    def will_player_win_after_n(self, state):
        """ 
        Return a boolean in regards to if the player will win the game or not 
        if they choose to stop playing the current round (i.e.: choose the 
        'n' action). 
        """
        won_columns_by_player = [won[0] for won in state.finished_columns if state.player_turn == won[1]]
        won_columns_by_player_this_round = [won[0] for won in state.player_won_column if state.player_turn == won[1]]
        if len(won_columns_by_player) + len(won_columns_by_player_this_round) >= 3:
            return True
        else:
            return False

    def will_player_win_after_n_teste(self,state):
        """ 
        Return a boolean in regards to if the player will win the game or not 
        if they choose to stop playing the current round (i.e.: choose the 
        'n' action). 
        """
        won_columns_by_player = [won[0] for won in state.finished_columns if state.player_turn == won[1]]
        won_columns_by_player_this_round = [won[0] for won in state.player_won_column if state.player_turn == won[1]]
        if len(won_columns_by_player) + len(won_columns_by_player_this_round) >= 3:
            return True
        else:
            return False

    def are_there_available_columns_to_play(self, state):
        """
        Return a booleanin regards to if there available columns for the player
        to choose. That is, if the does not yet have all three neutral markers
        used AND there are available columns that are not finished/won yet.
        """
        available_columns = self.get_available_columns(state)
        return state.n_neutral_markers != 3 and len(available_columns) > 0

    def get_available_columns(self, state):
        """ Return a list of all available columns. """

        # List containing all columns, remove from it the columns that are
        # available given the current board
        available_columns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for neutral in state.neutral_positions:
            available_columns.remove(neutral[0])
        for finished in state.finished_columns:
            if finished[0] in available_columns:
                available_columns.remove(finished[0])

        return available_columns

    def calculate_score(self, state, progress_value):
        score = 0
        neutrals = [col[0] for col in state.neutral_positions]
        for col in neutrals:
            advance = self.number_cells_advanced_this_round_for_col(state, col)
            # +1 because whenever a neutral marker is used, the weight of that
            # column is summed
            # Interporlated formula to find the array index given the column
            # y = 5-|x-7|
            score += (advance + 1) * progress_value[5 - abs(col - 7)]
        return score

    def number_cells_advanced_this_round(self, state):
        """
        Return the number of positions advanced in this round for current
        player for all columns.
        """
        counter = 0
        for column in range(state.column_range[0], state.column_range[1]+1):
            counter += self.number_cells_advanced_this_round_for_col(state, column)
        return counter

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

    def calculate_difficulty_score(self, state, odds, evens, highs, lows):
        """
        Add an integer to the current score given the peculiarities of the
        neutral marker positions on the board.
        """
        difficulty_score = 0

        neutral = [n[0] for n in state.neutral_positions]
        # If all neutral markers are in odd columns
        if all([x % 2 != 0 for x in neutral]):
            difficulty_score += odds
        # If all neutral markers are in even columns
        if all([x % 2 == 0 for x in neutral]):
            difficulty_score += evens
        # If all neutral markers are is "low" columns
        if all([x <= 7 for x in neutral]):
            difficulty_score += lows
        # If all neutral markers are is "high" columns
        if all([x >= 7 for x in neutral]):
            difficulty_score += highs

        return difficulty_score

    def advance(self, action):
        """ Return how many cells this action will advance for each column. """

        # Special case: doubled action (e.g. (6,6))
        if len(action) == 2 and action[0] == action[1]:
            return 2
        # All other cases will advance only one cell per column
        else:
            return 1

    def get_cols_weights(self, action):
        value = 0
        for col in action:
            if col in [2, 12]:
                value += 7
            elif col in [3, 11]:
                value += 0
            elif col in [4, 10]:
                value += 2
            elif col in [5, 9]:
                value += 0
            elif col in [6, 8]:
                value += 4
            else:
                value += 3
        return value

    def is_new_neutral(self, state, action):
        # Return a boolean representing if action will place a new neutral. """
        is_new_neutral = True
        for neutral in state.neutral_positions:
            if neutral[0] == action:
                is_new_neutral = False

        return is_new_neutral