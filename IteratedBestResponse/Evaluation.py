import sys
from DSL import *
from game import Game
from rule_of_28_sketch import Rule_of_28_Player_PS

class Evaluation:
    def eval(self):
        raise Exception('Unimplemented method: toString')
    
    def play_match(self, p1, p2):
        game = Game(n_players = 2, dice_number = 4, dice_value = 6, column_range = [2,12],
                    offset = 2, initial_height = 3)
        
        is_over = False
        who_won = None
    
        number_of_moves = 0
        current_player = game.player_turn
        while not is_over:
            moves = game.available_moves()
            if game.is_player_busted(moves):
                if current_player == 1:
                    current_player = 2
                else:
                    current_player = 1
                continue
            else:
                if game.player_turn == 1:
                    chosen_play = p1.get_action(game)
                else:
                    chosen_play = p2.get_action(game)
                if chosen_play == 'n':
                    if current_player == 1:
                        current_player = 2
                    else:
                        current_player = 1
                game.play(chosen_play)
                number_of_moves += 1
            who_won, is_over = game.is_finished()
            
            if is_over:
                return is_over, who_won
            
            if number_of_moves >= 300:
                return False, None

class FinishesGame(Evaluation):
    def __init__(self):
        self.program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
        self.program_decide_column = Argmax(Map(Function(Sum(Map(Function(Minus(Times(NumberAdvancedByAction(), VarScalarFromArray('move_value')), Times(VarScalar('marker'), IsNewNeutral()))), NoneNode()))), VarList('actions')))    
    
    def eval(self, program):
        p1 = Rule_of_28_Player_PS(self.program_yes_no, program)
        p2 = Rule_of_28_Player_PS(self.program_yes_no, program)
        
        number_matches_finished = 0
        total_number_matches = 50
        
        for i in range(total_number_matches):
            finished = False
            
            try:
                finished, _ = self.play_match(p1, p2)
                
                if finished:
                    number_matches_finished += 1
                
            except Exception:
                return False, True
        
        
        if number_matches_finished / total_number_matches > 0.6:
            return True, False
        
        return False, False
    
class DefeatsStrategy(Evaluation):
    def __init__(self, program):
        self.program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
        self.program_decide_column = Argmax(Map(Function(Sum(Map(Function(Minus(Times(NumberAdvancedByAction(), VarScalarFromArray('move_value')), Times(VarScalar('marker'), IsNewNeutral()))), NoneNode()))), VarList('actions')))
        
        self.player = Rule_of_28_Player_PS(self.program_yes_no, program)
        self.player_program = program   
        
    def play_n_matches(self, n, br):
        
        br_victories = 0
        player_victories = 0
        
        for _ in range(n):
            try:
                # plays a match with br as player 1
                finished, who_won = self.play_match(br, self.player)
                
                if finished:
                    if who_won == 1:
                        br_victories += 1
                    else:
                        player_victories += 1
                
                # plays another match with br as player 2        
                finished, who_won = self.play_match(self.player, br)
                
                if finished:
                    if who_won == 1:
                        player_victories += 1
                    else:
                        br_victories += 1
                        
            except Exception:
                return None, None, True
        
        return player_victories, br_victories, False
        
    def eval(self, program):
        
        if program.toString() == self.player_program.toString():
            return False, False
        
        br = Rule_of_28_Player_PS(self.program_yes_no, program)
        
        player_victories, br_victories, error = self.play_n_matches(5, br)
        
        if error:
            return False, True
        
        if br_victories + player_victories == 0:
            return False, False
        
        if (br_victories / (br_victories + player_victories)) < 0.20:
            return False, False
        
        player_victories_2, br_victories_2, error = self.play_n_matches(90, br)
        
        player_victories += player_victories_2
        br_victories += br_victories_2
        
        if br_victories + player_victories == 0:
            return False, False
        
        if (br_victories / (br_victories + player_victories)) < 0.55:
            return False, False

        player_victories_2, br_victories_2, error = self.play_n_matches(400, br)
        
        player_victories += player_victories_2
        br_victories += br_victories_2
        
        if br_victories + player_victories == 0:
            return False, False
        
        # consider a best response if defeats the other player in more than 55% of the matches
        if (br_victories / (br_victories + player_victories)) > 0.55:
            print('BR victories: ', br_victories)
            print('Player victories: ', player_victories)
            print()
            return True, False
        
        return False, False
    
class DefeatsStrategyNonTriage(Evaluation):
    def __init__(self, program):
        self.program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
        self.program_decide_column = Argmax(Map(Function(Sum(Map(Function(Minus(Times(NumberAdvancedByAction(), VarScalarFromArray('move_value')), Times(VarScalar('marker'), IsNewNeutral()))), NoneNode()))), VarList('actions')))
        
        self.player = Rule_of_28_Player_PS(self.program_yes_no, program)
        self.player_program = program   
        
    def play_n_matches(self, n, br):
        
        br_victories = 0
        player_victories = 0
        
        for _ in range(n):
            try:
                # plays a match with br as player 1
                finished, who_won = self.play_match(br, self.player)
                
                if finished:
                    if who_won == 1:
                        br_victories += 1
                    else:
                        player_victories += 1
                
                # plays another match with br as player 2        
                finished, who_won = self.play_match(self.player, br)
                
                if finished:
                    if who_won == 1:
                        player_victories += 1
                    else:
                        br_victories += 1
                        
            except Exception:
                return None, None, True
        
        return player_victories, br_victories, False
        
    def eval(self, program):
        
        if program.toString() == self.player_program.toString():
            return False, False
        
        br = Rule_of_28_Player_PS(self.program_yes_no, program)
        
        player_victories, br_victories, error = self.play_n_matches(500, br)
        
        if error:
            return False, True
        
        if br_victories + player_victories == 0:
            return False, False
        
        # consider a best response if defeats the other player in more than 55% of the matches
        if (br_victories / (br_victories + player_victories)) > 0.55:
            print('BR victories: ', br_victories)
            print('Player victories: ', player_victories)
            print()
            return True, False
        
        return False, False