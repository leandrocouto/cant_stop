import numpy as np

class Cell:
    def __init__(self):
        """A list of markers
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