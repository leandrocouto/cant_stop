import unittest
import sys
sys.path.insert(0,'..')

from game import Game
from MetropolisHastings.DSL import DSL 

class TestDSL(unittest.TestCase):
    def test_get_player_score(self):
        # Empty board
        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.get_player_score(a), 0)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.get_player_score(a), 15)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.get_player_score(a), 16)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [1], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [1]], 
                [[], []]
            ]
        finished_columns = [(3,1), (4,1)] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.get_player_score(a), 39)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []] 
            ]
        finished_columns = [(2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (9,1), (10,1), (11,1), (12,1)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.get_player_score(a), 83)
        
    def test_get_opponent_score(self):
        # Empty board
        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.get_opponent_score(a), 0)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.get_opponent_score(a), 4)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.get_opponent_score(a), 4)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [1], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [1]], 
                [[], []]
            ]
        finished_columns = [(3,1), (4,2)] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.get_opponent_score(a), 11)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []] 
            ]
        finished_columns = [(2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2), (9,2), (10,2), (11,2), (12,2)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.get_opponent_score(a), 83)

    def test_number_cells_advanced_this_round(self):
        # Empty board
        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.number_cells_advanced_this_round(a), 0)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.number_cells_advanced_this_round(a), 3)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.number_cells_advanced_this_round(a), 4)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[1], [], [], [], [0], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [1], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [1]], 
                [[], []]
            ]
        finished_columns = [(3,1), (5,2)] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.number_cells_advanced_this_round(a), 8)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []] 
            ]
        finished_columns = [(2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2), (9,2), (10,2), (11,2), (12,2)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(DSL.number_cells_advanced_this_round(a), 0)

    def test_advance_in_action_col(self):
        
        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.advance_in_action_col((2,2)), 2)

        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.advance_in_action_col((2,)), 1)

        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.advance_in_action_col((4,7)), 2)

    def test_does_action_place_new_neutral(self):
        # Empty board
        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.does_action_place_new_neutral((7,7), a), 1)

        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.does_action_place_new_neutral((7,), a), 1)

        a = Game(2, 4, 6, [2,12], 2, 2)
        self.assertEqual(DSL.does_action_place_new_neutral((4,7), a), 1)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.n_neutral_markers = 2
        a.neutral_positions.append((6, 4))
        a.neutral_positions.append((8, 9))
        self.assertEqual(DSL.does_action_place_new_neutral((6,6), a), 0)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.n_neutral_markers = 2
        a.neutral_positions.append((6, 4))
        a.neutral_positions.append((8, 9))
        self.assertEqual(DSL.does_action_place_new_neutral((6,7), a), 1)

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[1], [], [], [], [0], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [1], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [1]], 
                [[], []]
            ]
        finished_columns = [(3,1), (5,2)] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.n_neutral_markers = 3
        a.neutral_positions.append((4, 4))
        a.neutral_positions.append((6, 4))
        a.neutral_positions.append((8, 9))
        self.assertEqual(DSL.does_action_place_new_neutral((4,6), a), 0)


if __name__ == '__main__':
    unittest.main()
