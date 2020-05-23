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
                                'DSL.number_cells_advanced_this_round(state, COLS ) > SMALL_NUM',
                                'DSL.number_positions_conquered(state, COLS ) > SMALL_NUM',
                                'DSL.columns_won_by_opponent(state) > SMALL_NUM',
                                'DSL.columns_won_by_player(state) > SMALL_NUM'
                                ]
        #self._grammar['COLS'] = ['2', '3', '4', '5', '6']
        self._grammar['COLS'] = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        self._grammar['SMALL_NUM'] = ['0', '1', '2']
        
        self._reservedWords = ['if']

        self.rules = []

    @staticmethod
    def is_action_a_column_border(action):
        """ 
        Check if a column related to the action is located at the border of
        the board. For example, the action (2,6) would return True because 2
        is at the border (even though 6 is not). (6,5) would return False.
        """
        for column in action:
            if column in [2, 3, 11, 12]:
            #if column in [2, 6]:
                return True
        return False

    @staticmethod
    def is_doubled_action(action):
        """ Check if action is for example (6,6), (1,1). """
        if len(action) > 1 and action[0] == action[1]:
            return True
        else:
            return False
        
    @staticmethod    
    def is_column_in_action(action, column):
        """ 
        Check if column is in action. Might come in handy if the player focus
        on the border columns (or some combination with how far the player is
        in a certain column).
        """
        if not isinstance(action, str):
            if column in action:
                return True
        return False
    
    @staticmethod
    def action_wins_at_least_one_column(state, action):
        """ Check if action will result in at least one won column. """
        copy_state = state.clone()
        copy_state.play(action)
        columns_won = copy_state.player_won_column
        columns_won_previously = state.player_won_column
        if len(columns_won) > 0 and columns_won != columns_won_previously:
            return True
        return False
    
    @staticmethod
    def number_columns_won_current_round(state, action):
        """ Return how many columns current player won this round. """
        return len(state.player_won_column)
    
    @staticmethod
    def number_cells_advanced_this_round(state, column):
        """
        Return the number of positions advanced in this round for a given
        column by the player.
        """
        counter = 0
        previously_conquered = -1
        list_of_cells = state.board_game.board[column]
        
        for i in range(len(list_of_cells)):
            if state.player_turn in list_of_cells[i].markers:
                previously_conquered = i
            if 0 in list_of_cells[i].markers:
                counter = i - previously_conquered
                
        partial_completed_rows = [item[0] for item in state.player_won_column]
        if column in partial_completed_rows:
            counter += 1
        return counter
    
    @staticmethod
    def number_positions_conquered(state, column):
        """
        Return how far the player is in 'column'. -1 if the player is not in 
        the column.
        """
        previously_conquered = -1
        list_of_cells = state.board_game.board[column]
        
        for i in range(len(list_of_cells)):
            if state.player_turn in list_of_cells[i].markers:
                previously_conquered = i
           
        return previously_conquered

    @staticmethod
    def has_won_column_current_round(state, action):
        return len(state.player_won_column) > 0

    @staticmethod
    def columns_won_by_opponent(state):
        """ Return the number of columns won by the opponent. """
        columns_won = state.finished_columns
        current_player = state.player_turn
        if current_player == 1:
            opponent = 2
        else:
            opponent = 1
        count = 0
        for cols in columns_won:
            if cols[1] == opponent:
                count += 1
        return count

    @staticmethod
    def columns_won_by_player(state):
        """ Return the number of columns won by the player. """
        columns_won = state.finished_columns
        current_player = state.player_turn
        count = 0
        for cols in columns_won:
            if cols[1] == current_player:
                count += 1
        return count
    
    @staticmethod
    def is_stop_action(action):
        """ Check if the actions is 'n' (that is, a stop action). """
        if isinstance(action, str) and action == 'n':
            return True
        return False
