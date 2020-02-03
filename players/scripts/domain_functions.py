import copy

class DSL:
    @staticmethod
    def isDoubles(action):
        if len(action) > 1 and action[0] == action[1]:
            return True
        else:
            return False
        
    @staticmethod    
    def containsNumber(action, number):
        if not isinstance(action, str):
            if number in action:
                return True
        return False
    
    @staticmethod
    def actionWinsColumn(state, action):
        copy_state = copy.deepcopy(state)
        copy_state.play(action)
        columns_won = copy_state.columns_won_current_round()
        columns_won_previously = state.columns_won_current_round()
        if len(columns_won) > 0 and columns_won != columns_won_previously:
            return True
        return False
    
    @staticmethod
    def numberColumnsWon(state, action):
        return len(state.columns_won_current_round())