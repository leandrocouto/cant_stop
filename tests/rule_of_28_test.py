import unittest
import sys
sys.path.insert(0,'..')

from game import Game
from players.rule_of_28_player import Rule_of_28 

class TestRuleof28(unittest.TestCase):
    '''
                                 2  3  4  5  6  7  8  9 10 11 12
    self.progress_value = [0, 0, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6]
    self.move_value =     [0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    score += advance * (self.move_value[a]) - self.marker * markers
    '''
    def test_get_action(self):

        # Empty board
        a = Game(2, 4, 6, [2,12], 2, 2)
        rule_of_28 = Rule_of_28()
        a.current_roll = (1,1,1,2) #(2,3)
        self.assertEqual(rule_of_28.get_action(a), (2,3))

        a = Game(2, 4, 6, [2,12], 2, 2)
        rule_of_28 = Rule_of_28()
        a.current_roll = (1,2,3,4) # (3,7)->-4+0, (4,6)->-3-2, (5,5)->2
        self.assertEqual(rule_of_28.get_action(a), (5,5))

        a = Game(2, 4, 6, [2,12], 2, 2)
        rule_of_28 = Rule_of_28()
        a.current_roll = (2,3,4,5) # (5,9)->-2-2, (6,8)->-1-1, (7,7)->12-6
        self.assertEqual(rule_of_28.get_action(a), (7,7))

        a = Game(2, 4, 6, [2,12], 2, 2)
        rule_of_28 = Rule_of_28()
        a.current_roll = (2,3,4,5) # (5,9)->4-2, (6,8)->5-1, (7,7)->12-6
        manual_board = [
                [[], []], #2
                [[], [], [], []], #3 
                [[], [], [], [], [], []], #4 
                [[], [], [1], [], [], [0], [], []], #5 
                [[], [], [], [1,2], [], [], [], [], [], [0]], #6 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], #7 
                [[], [], [], [], [], [], [], [1], [], []], #8
                [[], [], [], [], [], [], [], []], #9
                [[], [1], [], [], [], [2]], #10
                [[], [], [], []], #11
                [[], []] #12
            ]
        finished_columns = [(12,1)] #(column, player)
        player_won_column = [(6, 2), (8, 1)]
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(rule_of_28.get_action(a), (7,7))

        # Article example
        a = Game(2, 4, 6, [2,12], 2, 2)
        rule_of_28 = Rule_of_28()
        a.current_roll = (5,2,3,2) # (7,5)-> 0+4, (8,)-> 5
        manual_board = [
                [[], []], #2
                [[], [], [], []], #3 
                [[], [], [], [], [], []], #4 
                [[], [], [1], [], [], [0], [], []], #5 
                [[], [], [], [1,2], [], [], [], [], [], []], #6 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], #7 
                [[], [], [], [], [], [], [], [1], [0], []], #8
                [[], [], [], [], [], [], [], []], #9
                [[], [1], [], [], [], [2]], #10
                [[], [], [], []], #11
                [[], []] #12
            ]
        finished_columns = [(4,1)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(rule_of_28.get_action(a), (8,))

        a = Game(2, 4, 6, [2,12], 2, 2)
        rule_of_28 = Rule_of_28()
        a.current_roll = (5,2,3,2) # (7,5)-> 0+4, (8,)-> 5
        manual_board = [
                [[], []], #2
                [[], [], [], []], #3 
                [[], [], [], [], [], []], #4 
                [[], [], [1], [], [], [0], [], []], #5 
                [[], [], [], [1,2], [], [], [], [], [], []], #6 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], #7 
                [[], [], [], [], [], [], [], [1], [0], []], #8
                [[], [], [], [], [], [], [], []], #9
                [[], [1], [], [], [], [2]], #10
                [[], [], [], []], #11
                [[], []] #12
            ]
        finished_columns = [(4,1)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(rule_of_28.get_action(a), (8,))

        # Empty board
        a = Game(2, 4, 6, [2,12], 2, 2)
        rule_of_28 = Rule_of_28()

        a.current_roll = (1,1,1,2) #(2,3)
        play = rule_of_28.get_action(a)
        a.play(play)
        self.assertEqual(play, (2,3))

        play = rule_of_28.get_action(a)
        a.play(play)
        self.assertEqual(play, 'y')

        a.current_roll = (1,2,3,4) #(3,7), (4,), (6,), (5,5)
        play = rule_of_28.get_action(a)
        a.play(play)
        self.assertEqual(play, (5,5))

        play = rule_of_28.get_action(a)
        a.play(play)
        self.assertEqual(play, 'n')


        a = Game(2, 4, 6, [2,12], 2, 2)
        rule_of_28 = Rule_of_28()
        a.current_roll = (5,2,3,2) # (7,5)-> 0+4, (8,)-> 5
        manual_board = [
                [[], []], #2
                [[], [], [], []], #3 
                [[], [], [], [], [], []], #4 
                [[], [], [1], [], [], [0], [], []], #5 
                [[], [], [], [1,2], [], [], [], [], [], []], #6 
                [[], [], [], [], [], [], [2], [], [], [], [1], []], #7 
                [[], [], [], [], [], [], [], [1], [0], []], #8
                [[], [], [], [], [], [], [], []], #9
                [[], [1], [], [], [], [2]], #10
                [[], [], [], []], #11
                [[], []] #12
            ]
        finished_columns = [(4,1)] #(column, player)
        player_won_column = []
        a.set_manual_board(manual_board, finished_columns, player_won_column)
        self.assertEqual(rule_of_28.get_action(a), (8,))


if __name__ == '__main__':
    unittest.main()