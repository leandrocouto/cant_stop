import numpy as np
import random

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
            if x < 7:
                offset += 2
            else:
                offset -= 2
    def print_board(self):
        for x in range(2,13):
            list_of_cells = self.board[x]
            print('{:3d}:'.format(x), end='')
            for cell in list_of_cells:
                print(cell.markers, end='')
            print()

class Game:
    def __init__(self, board_game = None, n_players = 2):
        """finished_columns is a list of 2-tuples indicating which columns were
        won by a certain player. Ex.: (2, 3) meaning column 2 won by player 4.
        player_won_column is a list of 2-tuples indicating which 
        columns were won by a certain player in the current round.
        Ex.: (2, 3) meaning column 2 won by player 4 in the current round.
        """
        self.board_game = Board()
        self.n_players = n_players
        self.finished_columns = []
        self.player_won_column = []
    def clone(self):
        return Game(self.board, self.n_players)
    def play(self, player_id, dice_combination):
        """player_id refers to either 1, 2, 3 or 4.
        dice_combination is a list with one or two integers representing the sum
        the player chose.
        """
        for die_position in range(len(dice_combination)):
            current_position_zero = 0
            current_position_id = -1
            row = dice_combination[die_position]
            cell_list = self.board_game.board[row]
            for i in range(0, len(cell_list)):
                if 0 in cell_list[i].markers:
                    current_position_zero = i
                if player_id in cell_list[i].markers:
                    current_position_id = i
            # If there's a zero in that column ahead of the player_id marker
            if current_position_id < current_position_zero:
                # If there's no zero and no player_id marker
                if current_position_zero == 0 and 0 not in cell_list[0].markers:
                    cell_list[current_position_zero].markers.append(0)
                else: # Zero is ahead of the player_id marker
                    #First check if the player will win that column
                    if current_position_zero == len(cell_list) - 1:
                        self.player_won_column.append((row, player_id))
                    else:
                        cell_list[current_position_zero].markers.remove(0)
                        cell_list[current_position_zero+1].markers.append(0)
            else: #There's no zero yet in that column
                #First check if the player will win that column
                if current_position_id == len(cell_list) - 1:
                    self.player_won_column.append((row, player_id))
                else:
                    cell_list[current_position_id+1].markers.append(0)
    def transform_neutral_markers(self, player_id):
        """Transform the neutral markers into player_id markers (1,2,3 or 4)."""
        for x in range(2,13):
            for i in range(len(self.board_game.board[x])):
                for j in range(len(self.board_game.board[x][i].markers)):
                    if self.board_game.board[x][i].markers[j] == 0:
                        self.board_game.board[x][i].markers[j] = player_id
        # Check if there are duplicates of player_id in the columns
        # If so, keep only the furthest one
        for x in range(2,13):
            list_of_cells = self.board_game.board[x]
            count = 0
            ocurrences_index = []
            for i in range(len(list_of_cells)):
                for j in range(len(list_of_cells[i].markers)):
                    if list_of_cells[i].markers[j] == player_id:
                        count += 1
                        ocurrences_index.append((x,i,j))
                       
            if count == 2:
                x, i, j = ocurrences_index[0]
                print('Tuplas = ', ocurrences_index[0], ocurrences_index[1])
                self.board_game.board[x][i].markers.remove(player_id)
        
        # Check if the player won some column and update it accordingly
        for column_won in self.player_won_column:
            self.finished_columns.append((column_won[0], column_won[1]))
            for cell in self.board_game.board[column_won[0]]:
                cell.markers.clear()
                cell.markers.append(player_id)
        self.player_won_column.clear()
        print('board no fim transform neutral')
        self.board_game.print_board()

    def erase_neutral_markers(self):
        """Remove the neutral markers because the player is busted."""
        for x in range(2,13):
            for i in range(len(self.board_game.board[x])):
                if 0 in self.board_game.board[x][i].markers:
                    self.board_game.board[x][i].markers.remove(0)
    def count_neutral_markers(self):
        """Return the number of neutral markers present in the current board."""
        count = 0
        for x in range(2,13):
            list_of_cells = self.board_game.board[x]
            for cell in list_of_cells:
                if 0 in cell.markers:
                    count += 1
        # Also has to take into account the columns the player won in the
        # current round.
        return count + len(self.player_won_column)
    def is_player_busted(self, all_moves):
        """Check if the player has no remaining play. Return a boolean.
        all_moves is a list of 2-tuples or integers relating to the possible
        plays the player can make.
        """
        if self.count_neutral_markers() < 3:
            return False
        for move in all_moves:
            for i in range(len(move)):
                list_of_cells = self.board_game.board[move[i]]
                for cell in list_of_cells:
                    if 0 in cell.markers:
                        return False
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
    def available_moves(self, dice):
        """Return a list of 2-tuples of possible combinations player_id can play
        if neutral counter is less than 2.
        dice is a 4-tuple with 4 integers representing the dice roll. Otherwise, 
        return a list of 2-tuples or integers according to the current board 
        schematic.
        """
        standard_combination = [(dice[0] + dice[1], dice[2] + dice[3]),
                                (dice[0] + dice[2], dice[1] + dice[3]),
                                (dice[0] + dice[3], dice[1] + dice[2])]
        print('Standard combination: ', standard_combination)
        if False:#self.count_neutral_markers() < 2:
            return standard_combination
        else:
            combination = []
            for comb in standard_combination:
                if self.check_tuple_availability(comb):
                    print('A tupla', comb, 'foi aceita.')
                    combination.append(comb)
                elif self.check_value_availability(comb[0]) and \
                          self.check_value_availability(comb[1]):
                    print('Os valores', comb[0], 'e', comb[1], 'foram aceitos separadamente.')
                    combination.append([comb[0]])
                    combination.append([comb[1]])
                if self.check_value_availability(comb[0]) and not\
                          self.check_value_availability(comb[1]):
                    print('O valor', comb[0], 'foi aceito.')
                    combination.append([comb[0]])
                if self.check_value_availability(comb[1]) and not\
                          self.check_value_availability(comb[0]):
                    print('O valor', comb[0], 'foi aceito.')
                    combination.append([comb[1]])
            return combination
    def is_finished(self):
    	won_columns_player_1 = 0
    	won_columns_player_2 = 0
    	for tuples in self.finished_columns:
    		if tuples[1] == 1:
    			won_columns_player_1 += 1
    		else:
    			won_columns_player_2 += 1
    	if won_columns_player_1 == 3 or won_columns_player_2 == 3:
    		return True
    	else:
    		return False
