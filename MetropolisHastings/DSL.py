class DSL:
    """
    Implementation of a Domain Specific Language (DSL) for the Can't Stop
    domain.
    """
    def __init__(self, start):
        self.start = start
        
        self._grammar = {}
        self._grammar[self.start] = [r"\t \t for a in actions : \n \t \t \t if a in ['y','n'] : \n \t \t \t \t forced_condition_if \n \t \t \t else : \n \t \t \t \t forced_condition_else"]
        self._grammar['forced_condition_if']   = [r"if BOOL_0 : \n \t \t \t \t \t return a \n \t \t \t \t condition_string"]
        self._grammar['forced_condition_else'] = [r"if BOOL_1 : \n \t \t \t \t \t return a \n \t \t \t \t condition_numeric"]
        self._grammar['condition_string'] = [r"if BOOL_0 : \n \t \t \t \t \t return a \n \t \t \t \t condition_string", ""]
        self._grammar['condition_numeric'] = [r"if BOOL_1 : \n \t \t \t \t \t return a \n \t \t \t \t condition_numeric", ""]
        self._grammar['BOOL_0'] = ["B_0", "B_0 and BOOL_0", "B_0 or BOOL_0"]
        self._grammar['BOOL_1'] = ["B_1", "B_1 and BOOL_1", "B_1 or BOOL_1"]
        self._grammar['B_0'] = [# Strictly "string" actions
                                'DSL.is_stop_action(a)',
                                # All types of actions
                                'DSL.get_player_score(state) > SCORE',
                                'DSL.get_opponent_score(state) > SCORE',
                                'DSL.number_of_neutral_markers_remaining(state) == SMALL_NUM',
                                'DSL.number_cells_advanced_this_round(state) > COLS',
                                'DSL.columns_won_by_opponent(state) == SMALL_NUM',
                                'DSL.columns_won_by_player(state) == SMALL_NUM',
                                'DSL.has_won_column_current_round(state)'
                                ]
        self._grammar['B_1'] = [# Strictly "numeric" actions
                                'DSL.is_doubled_action(a)', 
                                'DSL.is_action_a_column_border(a)', 
                                'DSL.has_won_column_current_round(state)', 
                                'DSL.is_column_in_action(a, COLS )',
                                'DSL.action_wins_at_least_one_column(state,a)',
                                # All types of actions
                                'DSL.number_cells_advanced_this_round_for_col(state, COLS ) > SMALL_NUM',
                                'DSL.number_positions_conquered(state, COLS ) > SMALL_NUM',
                                'DSL.columns_won_by_opponent(state) == SMALL_NUM',
                                'DSL.columns_won_by_player(state) == SMALL_NUM',
                                'DSL.number_of_neutral_markers_remaining(state) == SMALL_NUM',
                                'DSL.number_cells_advanced_this_round(state) > COLS',
                                'DSL.get_player_score(state) > SCORE',
                                'DSL.get_opponent_score(state) > SCORE'
                                ]
        self._grammar['SCORE'] = ['5', '10', '15', '20', '30', '40', '50', '60', '70']
        self._grammar['COLS'] = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        self._grammar['SMALL_NUM'] = ['0', '1', '2', '3']

    @staticmethod
    def get_player_score(state):
        """ A score is the number of cells advanced in all columns. """

        counter = 0
        player = state.player_turn
        won = []
        # First look for columns already won and sum it accordingly
        for won_column in state.finished_columns:
            won.append(won_column[0])
            if won_column[1] == player:
                counter += len(state.board_game.board[won_column[0]]) + 1
        # Now sum for the remaining columns
        for column in range(state.column_range[0], state.column_range[1]+1):
            # Ignore columns alread won, they are already counted above
            if column in won:
                continue
            previously_conquered = -1
            neutral_position = -1
            list_of_cells = state.board_game.board[column]

            for i in range(len(list_of_cells)):
                if player in list_of_cells[i].markers:
                    previously_conquered = i
                if 0 in list_of_cells[i].markers:
                    neutral_position = i
            if neutral_position != -1:
                counter += neutral_position + 1
                for won_column in state.player_won_column:
                    if won_column[0] == column:
                        counter += 1
            elif previously_conquered != -1 and neutral_position == -1:
                counter += previously_conquered + 1
                for won_column in state.player_won_column:
                    if won_column[0] == column:
                        counter += len(list_of_cells) - previously_conquered
        return counter

    @staticmethod
    def get_opponent_score(state):
        """ A score is the number of cells advanced in all columns. """

        counter = 0
        if state.player_turn == 1:
            player = 2
        else:
            player = 1
        won = []
        # First look for columns already won and sum it accordingly
        for won_column in state.finished_columns:
            won.append(won_column[0])
            if won_column[1] == player:
                counter += len(state.board_game.board[won_column[0]]) + 1
        # Now sum for the remaining columns
        for column in range(state.column_range[0], state.column_range[1]+1):
            # Ignore columns alread won, they are already counted above
            if column in won:
                continue
            previously_conquered = -1
            list_of_cells = state.board_game.board[column]

            for i in range(len(list_of_cells)):
                if player in list_of_cells[i].markers:
                    previously_conquered = i
            if previously_conquered != -1:
                counter += previously_conquered + 1
        return counter

    @staticmethod
    def number_of_neutral_markers_remaining(state):
        """ Return how many neutral markers the player has in their disposal."""
        return 3 - state.n_neutral_markers
    
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
    def number_cells_advanced_this_round_for_col(state, column):
        """
        Return the number of positions advanced in this round for a given
        column by the player.
        """
        counter = 0
        previously_conquered = -1
        neutral_position = -1
        list_of_cells = state.board_game.board[column]

        for i in range(len(list_of_cells)):
            if state.player_turn in list_of_cells[i].markers:
                previously_conquered = i
            if 0 in list_of_cells[i].markers:
                neutral_position = i
        if previously_conquered == -1 and neutral_position != -1:
            counter += neutral_position + 1
            for won_column in state.player_won_column:
                if won_column[0] == column:
                    counter += 1
        elif previously_conquered != -1 and neutral_position != -1:
            counter += neutral_position - previously_conquered
            for won_column in state.player_won_column:
                if won_column[0] == column:
                    counter += 1
        elif previously_conquered != -1 and neutral_position == -1:
            for won_column in state.player_won_column:
                if won_column[0] == column:
                    counter += len(list_of_cells) - previously_conquered
        return counter

    @staticmethod
    def number_cells_advanced_this_round(state):
        """
        Return the number of positions advanced in this round for current
        player for all columns.
        """
        counter = 0
        for column in range(state.column_range[0], state.column_range[1]+1):
            counter += DSL.number_cells_advanced_this_round_for_col(state, column)
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
    def has_won_column_current_round(state):
        """ Check if player has won any column in the current round. """
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
