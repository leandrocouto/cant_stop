import copy

class DSL:
    """
    Implementation of a Domain Specific Language (DSL) for the Can't Stop
    domain.
    """
    def __init__(self):
        self.start = 'S'
        
        self._grammar = {}
        self._grammar[self.start] = ['if BOOL OP S', '']
        self._grammar['BOOL'] = ['B_0', 'B_1']
        self._grammar['OP'] = ['and BOOL OP', 'or BOOL OP', '']
        self._grammar['B_0'] = ['DSL.is_doubled_action(a)', 
                                'DSL.action_wins_at_least_one_column(state,a)', 
                                'DSL.has_won_column_current_round(state,a)', 
                                'DSL.is_stop_action(a)',
                                'DSL.is_action_a_column_border(a)'
                                ]
        self._grammar['B_1'] = [
                                'DSL.is_column_in_action(a, COLS )',
                                'DSL.number_positions_conquered_this_round(state, COLS ) > SMALL_NUM',
                                'DSL.number_positions_conquered(state, COLS ) > SMALL_NUM'
                                ]
        #self._grammar['COLS'] = ['2', '3', '4', '5', '6']
        self._grammar['COLS'] = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        self._grammar['SMALL_NUM'] = ['0', '1', '2']
        
        self._reservedWords = ['if']

        self.rules = []

    @staticmethod
    def is_action_a_column_border(action):
        for column in action:
            if column in [2, 3, 11, 12]:
            #if column in [2, 6]:
                return True
        return False

    @staticmethod
    def is_doubled_action(action):
        if len(action) > 1 and action[0] == action[1]:
            return True
        else:
            return False
        
    @staticmethod    
    def is_column_in_action(action, column):
        if not isinstance(action, str):
            if column in action:
                return True
        return False
    
    @staticmethod
    def action_wins_at_least_one_column(state, action):
        copy_state = copy.deepcopy(state)
        copy_state.play(action)
        columns_won = copy_state.columns_won_current_round()
        columns_won_previously = state.columns_won_current_round()
        if len(columns_won) > 0 and columns_won != columns_won_previously:
            return True
        return False
    
    @staticmethod
    def number_columns_won_current_round(state, action):
        return len(state.columns_won_current_round())
    
    @staticmethod
    def number_positions_conquered_this_round(state, column):
        return state.number_positions_conquered_this_round(column)
    
    @staticmethod
    def number_positions_conquered(state, column):
        return state.number_positions_conquered(column)

    @staticmethod
    def has_won_column_current_round(state, action):
        return len(state.columns_won_current_round()) > 0
    
    @staticmethod
    def is_stop_action(action):
        if isinstance(action, str) and action == 'n':
            return True
        return False
