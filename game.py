import numpy as np
import random
import copy

class Cell:
    def __init__(self):
        """A list of markers.
        Neutral markers are represented as 0.
        Markers from Player 1 are represented as 1, and so forth.
        """
        self.markers = []

class Board:
    def __init__(self):
        """First two columns are unused.
        Used columns vary from range 2 to 12 (inclusive).
        """
        self.board = [[] for _ in range(13)]
        offset = 2
        for x in range(2,13):
            for _ in range(offset):
                self.board[x].append(Cell())
            # This empty string will be replaces by a '*' if the row
            # has been partially completed (i.e.: the player completed that row
            # but their turn has not ended just yet).
            # self.board[x].append('')
            if x < 7:
                offset += 2
            else:
                offset -= 2
    def print_board(self, rows):
        partial_completed_rows = [item[0] for item in rows]
        for x in range(2,13):
            list_of_cells = self.board[x]
            print('{:3d}:'.format(x), end='')
            for cell in list_of_cells:
                print(cell.markers, end='')
            if x in partial_completed_rows:
                print('*', end='')
            print()

class Game:
    def __init__(self, board_game = None, n_players = 2, finished_columns = None,
                player_won_column = None, player_turn = None, dice_action = None,
                current_roll = None):
        """finished_columns is a list of 2-tuples indicating which columns were
        won by a certain player. Ex.: (2, 3) meaning column 2 won by player 3.
        player_won_column is a list of 2-tuples indicating which 
        columns were won by a certain player in the current round.
        Ex.: (2, 3) meaning column 2 won by player 3 in the current round.
        dice_action refers to which action the game is at at the moment, if the
        player is choosing which combination or if the player is choosing if he
        wants to continue playing the turn or not.
        """
        self.board_game = board_game or Board()
        self.n_players = n_players
        self.finished_columns = finished_columns or []
        self.player_won_column = player_won_column or []
        self.player_turn = player_turn or 1
        self.dice_action = dice_action or True
        self.current_roll = current_roll or self.roll_dice()
    def clone(self):
        return copy.deepcopy(self)
        #return Game(self.board_game, self.n_players, self.finished_columns,
        #            self.player_won_column, self.player_turn, self.dice_action,
        #            self.current_roll)
    def play(self, chosen_play):
        """player_id refers to either 1, 2, 3 or 4.
        dice_combination is a list with one or two integers representing the sum
        the player chose.
        """
        if chosen_play == 'n':
            self.transform_neutral_markers()
            # Next action should be to choose a dice combination
            self.dice_action = True
            return
        if chosen_play == 'y':
            # Next action should be to choose a dice combination
            self.dice_action = True
            return
        if self.is_player_busted(self.available_moves()):
            return
        for die_position in range(len(chosen_play)):
            current_position_zero = 0
            current_position_id = -1
            row = chosen_play[die_position]
            cell_list = self.board_game.board[row]
            for i in range(0, len(cell_list)):
                if 0 in cell_list[i].markers:
                    current_position_zero = i
                if self.player_turn in cell_list[i].markers:
                    current_position_id = i
            # If there's a zero in that column ahead of the player_id marker
            if current_position_id < current_position_zero:
                # If there's no zero and no player_id marker
                if current_position_zero == 0 and 0 not in cell_list[0].markers:
                    cell_list[current_position_zero].markers.append(0)
                else: # Zero is ahead of the player_id marker
                    #First check if the player will win that column
                    if current_position_zero == len(cell_list) - 1:
                        self.player_won_column.append((row, self.player_turn))
                    else:
                        cell_list[current_position_zero].markers.remove(0)
                        cell_list[current_position_zero+1].markers.append(0)
            else: #There's no zero yet in that column
                #First check if the player will win that column
                if current_position_id == len(cell_list) - 1:
                    self.player_won_column.append((row, self.player_turn))
                else:
                    cell_list[current_position_id+1].markers.append(0)
        # Next action should be to choose if the player wants to continue or not
        self.dice_action = False
        # Then a new dice roll is done (same is done if the player is busted)
        self.current_roll = self.roll_dice()
    def simulated_play(self, chosen_play):
        """player_id refers to either 1, 2, 3 or 4.
        dice_combination is a list with one or two integers representing the sum
        the player chose.
        """
        simulated_game = self.clone()
        if chosen_play == 'n':
            simulated_game.transform_neutral_markers()
            return simulated_game.board_game
        if chosen_play == 'y':
            return simulated_game.board_game
        if simulated_game.is_player_busted(simulated_game.available_moves()):
            return simulated_game.board_game
        for die_position in range(len(chosen_play)):
            current_position_zero = 0
            current_position_id = -1
            row = chosen_play[die_position]
            cell_list = simulated_game.board_game.board[row]
            for i in range(0, len(cell_list)):
                if 0 in cell_list[i].markers:
                    current_position_zero = i
                if simulated_game.player_turn in cell_list[i].markers:
                    current_position_id = i
            # If there's a zero in that column ahead of the player_id marker
            if current_position_id < current_position_zero:
                # If there's no zero and no player_id marker
                if current_position_zero == 0 and 0 not in cell_list[0].markers:
                    cell_list[current_position_zero].markers.append(0)
                else: # Zero is ahead of the player_id marker
                    #First check if the player will win that column
                    if current_position_zero == len(cell_list) - 1:
                        simulated_game.player_won_column.append((row, 
                                                    simulated_game.player_turn))
                    else:
                        cell_list[current_position_zero].markers.remove(0)
                        cell_list[current_position_zero+1].markers.append(0)
            else: #There's no zero yet in that column
                #First check if the player will win that column
                if current_position_id == len(cell_list) - 1:
                    simulated_game.player_won_column.append((row, 
                                                    simulated_game.player_turn))
                else:
                    cell_list[current_position_id+1].markers.append(0)
        # Next action should be to choose if the player wants to continue or not
        simulated_game.dice_action = False
        # Then a new dice roll is done (same is done if the player is busted)
        simulated_game.current_roll = simulated_game.roll_dice()
        return simulated_game.board_game
    def transform_neutral_markers(self):
        """Transform the neutral markers into player_id markers (1,2,3 or 4)."""
        for x in range(2,13):
            for i in range(len(self.board_game.board[x])):
                for j in range(len(self.board_game.board[x][i].markers)):
                    if self.board_game.board[x][i].markers[j] == 0:
                        self.board_game.board[x][i].markers[j] = self.player_turn
        # Check if there are duplicates of player_id in the columns
        # If so, keep only the furthest one. Must make sure to not check
        # already won columns.
        completed_rows = [item[0] for item in self.finished_columns]
        for x in range(2,13):
            if x in completed_rows:
                continue
            list_of_cells = self.board_game.board[x]
            count = 0
            ocurrences_index = []
            for i in range(len(list_of_cells)):
                for j in range(len(list_of_cells[i].markers)):
                    if list_of_cells[i].markers[j] == self.player_turn:
                        count += 1
                        ocurrences_index.append((x,i,j))
                       
            if count == 2:
                x, i, _ = ocurrences_index[0]
                self.board_game.board[x][i].markers.remove(self.player_turn)
        
        # Check if the player won some column and update it accordingly.

        # Special case: Player 1 is about to win, for ex. column 7 but he rolled
        # a (7,7) tuple. That would add two instances (1,7) in the finished
        # columns list. Remove the duplicates from player_won_column.
        self.player_won_column = list(set(self.player_won_column))

        for column_won in self.player_won_column:
            self.finished_columns.append((column_won[0], column_won[1]))
            for cell in self.board_game.board[column_won[0]]:
                cell.markers.clear()
                cell.markers.append(self.player_turn)
        self.player_won_column.clear()

        if self.player_turn == self.n_players:
                self.player_turn = 1
        else:
            self.player_turn += 1

    def erase_neutral_markers(self):
        """Remove the neutral markers because the player is busted."""
        for x in range(2,13):
            for i in range(len(self.board_game.board[x])):
                if 0 in self.board_game.board[x][i].markers:
                    self.board_game.board[x][i].markers.remove(0)
    def count_neutral_markers(self):
        """Return the number of neutral markers present in the current board."""
        count = 0
        # Has to take into account the columns the player won in the current
        # round.
        partial_completed_rows = [item[0] for item in self.player_won_column]
        for row in partial_completed_rows:
            for cell in self.board_game.board[row]:
                if 0 in cell.markers:
                    count += 1
        # Iterate through the other rows.
        for x in range(2,13):
            if x in partial_completed_rows:
                continue
            list_of_cells = self.board_game.board[x]
            for cell in list_of_cells:
                if 0 in cell.markers:
                    count += 1
        
        return count
    def is_player_busted(self, all_moves):
        """Check if the player has no remaining play. Return a boolean.
        all_moves is a list of 2-tuples or integers relating to the possible
        plays the player can make or the [y,n] list regarding the turn the
        player chooses if he wants to continue playing or not.
        """
        if all_moves == ['y', 'n']:
        	return False
        if len(all_moves) == 0:
            self.erase_neutral_markers()
            self.player_won_column.clear()
            if self.player_turn == self.n_players:
                self.player_turn = 1
            else:
                self.player_turn += 1
            # Then a new dice roll is done (same is done if a play is completed)
            self.current_roll = self.roll_dice()
            return True
        if self.count_neutral_markers() < 3:
            return False
        for move in all_moves:
            for i in range(len(move)):
                list_of_cells = self.board_game.board[move[i]]
                for cell in list_of_cells:
                    if 0 in cell.markers:
                        return False
        self.erase_neutral_markers()
        self.player_won_column.clear()
        if self.player_turn == self.n_players:
            self.player_turn = 1
        else:
            self.player_turn += 1
        # Then a new dice roll is done (same is done if a play is completed)
        self.current_roll = self.roll_dice()
        return True
    def roll_dice(self):
        """Return a 4-tuple with four integers representing the dice roll."""
        return (random.randrange(1,7), random.randrange(1,7), 
                random.randrange(1,7), random.randrange(1,7))
    def check_tuple_availability(self, tuple):
        """Return a boolean.
        Check if there is a neutral marker in both tuples columns taking into
        account the number of neutral_markers currently on the board.
        """
        #First check if the column 'value' is already completed
        for tuple_finished in self.finished_columns:
            if tuple_finished[0] == tuple[0] or tuple_finished[0] == tuple[1]:
                return False
        for tuple_column in self.player_won_column:
            if tuple_column[0] == tuple[0] or tuple_column[0] == tuple[1]:
                return False
        is_first_value_valid = False
        is_second_value_valid = False
        neutral_markers = self.count_neutral_markers()
        for cell in self.board_game.board[tuple[0]]:
            if 0 in cell.markers:
                is_first_value_valid = True
        
        for cell in self.board_game.board[tuple[1]]:
            if 0 in cell.markers:
                is_second_value_valid = True
        if not is_first_value_valid and not is_second_value_valid and \
                neutral_markers == 0:
            return True  
        elif is_first_value_valid and is_second_value_valid:
            return True
        elif is_first_value_valid and not is_second_value_valid and \
                neutral_markers == 2:
            return True
        elif is_first_value_valid and not is_second_value_valid and \
                neutral_markers == 3:
            return False
        elif not is_first_value_valid and is_second_value_valid and \
                neutral_markers == 2:
            return True
        elif not is_first_value_valid and is_second_value_valid and \
                neutral_markers == 3:
            return False
        elif not is_first_value_valid and not is_second_value_valid and \
                neutral_markers == 2 and tuple[0] == tuple[1]:
            return True
        elif not is_first_value_valid and not is_second_value_valid and \
                neutral_markers == 1:
            return True
        else:
            return False
    def check_value_availability(self, value):
        """Return a boolean. 
        Check if there's a neutral marker in the 'value' column.
        """
        #First check if the column 'value' is already completed
        for tuple_finished in self.finished_columns:
            if tuple_finished[0] == value:
                return False
        for tuple_column in self.player_won_column:
            if tuple_column[0] == value:
                return False
        
        if self.count_neutral_markers() < 3:
            return True
        list_of_cells = self.board_game.board[value]
        for cell in list_of_cells:
            if 0 in cell.markers:
                return True
        return False
    def available_moves(self):
        """Return a list of 2-tuples of possible combinations player_id can play
        if neutral counter is less than 2 or return the list [y,n] in case the
        current action is to continue to play or not.
        dice is a 4-tuple with 4 integers representing the dice roll. Otherwise, 
        return a list of 2-tuples or integers according to the current board 
        schematic.
        """
        if not self.dice_action:
            return ['y','n']
        standard_combination = [(self.current_roll[0] + self.current_roll[1], 
        						self.current_roll[2] + self.current_roll[3]),
                                (self.current_roll[0] + self.current_roll[2], 
                                self.current_roll[1] + self.current_roll[3]),
                                (self.current_roll[0] + self.current_roll[3], 
                                self.current_roll[1] + self.current_roll[2])]
        #print('Standard combination: ', standard_combination)
        combination = []
        for comb in standard_combination:
            if self.check_tuple_availability(comb):
                combination.append(comb)
            elif self.check_value_availability(comb[0]) and \
                      self.check_value_availability(comb[1]):
                combination.append((comb[0],))
                combination.append((comb[1],))
            if self.check_value_availability(comb[0]) and not\
                      self.check_value_availability(comb[1]):
                combination.append((comb[0],))
            if self.check_value_availability(comb[1]) and not\
                      self.check_value_availability(comb[0]):
                combination.append((comb[1],))
        return combination
    def is_finished(self):
    	won_columns_player_1 = 0
    	won_columns_player_2 = 0
    	for tuples in self.finished_columns:
    		if tuples[1] == 1:
    			won_columns_player_1 += 1
    		else:
    			won_columns_player_2 += 1
        # >= because the player can have 2 columns and win another 2 columns
        # in one turn.
    	if won_columns_player_1 >= 3:
    		return 1, True
    	elif  won_columns_player_2 >= 3:
    		return 2, True
    	else:
    		return -1, False
