import unittest
import sys
sys.path.insert(0,'..')

from game import Game

class TestGame(unittest.TestCase):
    def test_board_game_equality(self):
        # Empty board
        a = Game(2, 4, 6, [2,12], 2, 2)
        b = Game(2, 4, 6, [2,12], 2, 2)
        self.assertTrue(a.check_boardgame_equality(b))
        self.assertTrue(b.check_boardgame_equality(a))

        # Empty and nonempty board
        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[0], [1]], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [1], [0], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [2,1], [0]], 
                [[], []]
            ]
        finished_columns = [(6,2), (5,2), (3,1), (12, 1)] #(column, player)
        player_won_column = [(9, 1), (10, 1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertFalse(a.check_boardgame_equality(b))
        self.assertFalse(b.check_boardgame_equality(a))

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[1], []], 
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
        finished_columns = [] #(column, player)
        player_won_column = []
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertFalse(a.check_boardgame_equality(b))
        self.assertFalse(b.check_boardgame_equality(a))

        b = Game(2, 4, 6, [2,12], 2, 2)
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
        finished_columns = [(3,1)] #(column, player)
        player_won_column = []
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertFalse(a.check_boardgame_equality(b))
        self.assertFalse(b.check_boardgame_equality(a))

        b = Game(2, 4, 6, [2,12], 2, 2)
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
        finished_columns = [] #(column, player)
        player_won_column = [(3,1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertFalse(a.check_boardgame_equality(b))
        self.assertFalse(b.check_boardgame_equality(a))

        #Non empty boards
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
        finished_columns = [] #(column, player)
        player_won_column = [(3,1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)

        b = Game(2, 4, 6, [2,12], 2, 2)
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
        finished_columns = [] #(column, player)
        player_won_column = [(3,1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertTrue(a.check_boardgame_equality(b))
        self.assertTrue(b.check_boardgame_equality(a))

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
        finished_columns = [] #(column, player)
        player_won_column = [(3,1), (4, 2)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)

        b = Game(2, 4, 6, [2,12], 2, 2)
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
        finished_columns = [] #(column, player)
        player_won_column = [(4,2), (3,1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertTrue(a.check_boardgame_equality(b))
        self.assertTrue(b.check_boardgame_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = [(3,1), (4, 2)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = [(4,2), (3,1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertTrue(a.check_boardgame_equality(b))
        self.assertTrue(b.check_boardgame_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = [(3,1), (4, 2)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [] #(column, player)
        player_won_column = [(4,2), (3,1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertFalse(a.check_boardgame_equality(b))
        self.assertFalse(b.check_boardgame_equality(a))

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
        player_won_column = [(3,1), (4, 2)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(8,2)] #(column, player)
        player_won_column = [(4,2), (3,1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertFalse(a.check_boardgame_equality(b))
        self.assertFalse(b.check_boardgame_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [], [2]], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [], [2]], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertTrue(a.check_boardgame_equality(b))
        self.assertTrue(b.check_boardgame_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [], [2]], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 1)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertFalse(a.check_boardgame_equality(b))
        self.assertFalse(b.check_boardgame_equality(a))

    def test_play(self):
        # Play after empty board

        a = Game(2, 4, 6, [2,12], 2, 2)
        b = Game(2, 4, 6, [2,12], 2, 2)
        b.play((7,7))
        self.assertFalse(a.check_game_equality(b))
        self.assertFalse(b.check_game_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        b = Game(2, 4, 6, [2,12], 2, 2)
        b.play((7,7))
        b.play('y')
        self.assertFalse(a.check_game_equality(b))
        self.assertFalse(b.check_game_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        b = Game(2, 4, 6, [2,12], 2, 2)
        b.play((7,7))
        b.play('y')
        b.play((2,4))
        b.play('n')
        self.assertFalse(a.check_game_equality(b))
        self.assertFalse(b.check_game_equality(a))

        #Not empty boards

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(8, 2)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.player_turn = 2
        a.n_neutral_markers = 2
        a.neutral_positions.append((6, 9))
        a.neutral_positions.append((8, 9))
        a.play((6,6))

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 2)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        b.player_turn = 2
        b.n_neutral_markers = 2
        b.neutral_positions.append((6, 9))
        b.neutral_positions.append((8, 9))

        self.assertTrue(a.check_game_equality(b))
        self.assertTrue(b.check_game_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [0], []], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(8, 2)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.player_turn = 2
        a.n_neutral_markers = 2
        a.neutral_positions.append((6, 8))
        a.neutral_positions.append((8, 9))
        a.play((6,6))

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 2)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        b.player_turn = 2
        b.n_neutral_markers = 2
        b.neutral_positions.append((6, 9))
        b.neutral_positions.append((8, 9))

        self.assertTrue(a.check_game_equality(b))
        self.assertTrue(b.check_game_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.player_turn = 2
        a.n_neutral_markers = 2
        a.neutral_positions.append((6, 9))
        a.neutral_positions.append((8, 9))
        a.play((6,8))

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 2)]
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        b.player_turn = 2
        b.n_neutral_markers = 2
        b.neutral_positions.append((6, 9))
        b.neutral_positions.append((8, 9))

        self.assertTrue(a.check_game_equality(b))
        self.assertTrue(b.check_game_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 2)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.player_turn = 2
        a.n_neutral_markers = 2
        a.neutral_positions.append((6, 9))
        a.neutral_positions.append((8, 9))
        a.play('n')

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1), (6, 2), (8, 2)] #(column, player)
        player_won_column = []
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        b.player_turn = 1
        b.n_neutral_markers = 0

        self.assertTrue(a.check_game_equality(b))
        self.assertTrue(b.check_game_equality(a))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [0], [], [], [], [], []], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [0], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], [0]], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.player_turn = 1
        a.n_neutral_markers = 3
        a.neutral_positions.append((6, 4))
        a.neutral_positions.append((8, 8))
        a.neutral_positions.append((10, 5))
        a.play('n')

        b = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [2], [1], [], [], [], [], []], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [2], [1]], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = []
        b.set_manual_board(manual_board, finished_columns, player_won_column)
        b.player_turn = 2
        b.n_neutral_markers = 0

        self.assertTrue(a.check_game_equality(b))
        self.assertTrue(b.check_game_equality(a))

    def test_available_moves(self):
        # Empty board
        a = Game(2, 4, 6, [2,12], 2, 2)
        a.current_roll = (1,1,1,1)
        self.assertEqual(set(a.available_moves()), set([(2,2)]))

        a = Game(2, 4, 6, [2,12], 2, 2)
        a.current_roll = (1,2,3,4)
        self.assertEqual(set(a.available_moves()), set([(3,7), (4, 6), (5, 5)]))

        a = Game(2, 4, 6, [2,12], 2, 2)
        a.play((2,2))
        self.assertEqual(set(a.available_moves()), set(['y', 'n']))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.current_roll = (6,6,4,4)
        self.assertEqual(set(a.available_moves()), set([(10,10)]))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1), (8,2), (10,1)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.current_roll = (6,6,4,4)
        self.assertEqual(set(a.available_moves()), set([]))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1), (8,2), (10,1)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.current_roll = (6,6,6,6)
        self.assertEqual(set(a.available_moves()), set([]))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.current_roll = (6,6,2,2)
        self.assertEqual(set(a.available_moves()), set([(4,)]))

        a = Game(2, 4, 6, [2,12], 2, 2)
        manual_board = [
                [[], []], 
                [[], [], [], []], 
                [[], [], [], [], [], []], 
                [[], [], [], [], [], [], [], []], 
                [[], [], [], [1,2], [], [], [], [], [], [0]], 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], 
                [[], [], [], [], [], [], [], [1], [], [0]], 
                [[], [], [], [], [], [], [], []], 
                [[], [1], [], [], [2], []], 
                [[], [], [], []], 
                [[], []]
            ]
        finished_columns = [(11,1)] #(column, player)
        player_won_column = [(8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        a.current_roll = (3,4,5,6)
        self.assertEqual(set(a.available_moves()), set([(7,), (10,), (9,9)]))


if __name__ == '__main__':
    unittest.main()
