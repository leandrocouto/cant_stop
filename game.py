import numpy as np

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
            offset += 2

class Game:
    def __init__(self, board = None, n_players):
        self.board = board or Board()
        self.n_players = n_players
    def clone(self):
        return Game(self.board, self.n_players)
    def update_board(self, player_id, dice_combination):
        """player_id refers to either 1, 2, 3 or 4
        dice_combination is 2-tuple chosen by the player
        """
        print()
    def is_player_busted(self, player_id, dice):
        """player_id refers to either 1, 2, 3 or 4
        dice is a 2-tuple representing the dice values
        Returns a boolean
        """
        print()
    def roll_dice(self):
        return (random.randrange(7), random.randrange(7))
    def get_possible_moves(self, player_id, dice):
        """player_id refers to either 1, 2, 3 or 4
        dice is a 2-tuple representing the dice values
        Returns a list of possible combinations player_id can play 
        based on the current board schematic
        """
        print()