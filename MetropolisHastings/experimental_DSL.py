class ExperimentalDSL:
    """
    Implementation of a Domain Specific Language (DSL) for the Can't Stop
    domain.
    """
    def __init__(self, start):
        
        self.start = start
        
        self._grammar = {}
        
        self._grammar['header'] = [
                                    r"progress_value=[0,0, INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS ]" +
                                    r"\n\t\tmove_value=[0,0, INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS , INTEGERS ]" +
                                    r"\n\t\todds= INTEGERS " +
                                    r"\n\t\tevens= INTEGERS " +
                                    r"\n\t\thighs= INTEGERS " +
                                    r"\n\t\tlows= INTEGERS " +
                                    r"\n\t\tmarker= INTEGERS " +
                                    r"\n\t\tthreshold= INTEGERS"
                                    ]
        self._grammar['body_if']   = [
                                        r"if actions[0] in ['y','n']: \n\t\t\tif bool_template :\n\t\t\t\treturn 'y'\n\t\t\telse:\n\t\t\t\treturn 'n' "
                                        ]
        self._grammar['body_else']   = [
                                        r"else:" +
                                        r"\n\t\t\tscores=np.zeros(len(actions))" +
                                        r"\n\t\t\tfor i in range(len(scores)):" +
                                        r"\n\t\t\t\tfor column in actions[i]:" +
                                        r"\n\t\t\t\t\tscores[i]+= expression_col " +
                                        r"\n\t\t\tchosen_action=actions[np.argmax(scores)] " +
                                        r"\n\t\t\treturn chosen_action"
                                        ]

        self._grammar['bool_template'] = [
                                    "bool_exp",
                                    "bool_exp BOOL_OP bool_exp",
                                    "bool_exp BOOL_OP bool_template"
        ]

        self._grammar['bool_exp'] = [
                                    "functions_bool",
                                    "functions_yes_no COMP_OP INTEGERS",
                                    "functions_yes_no COMP_OP variable",
                                    "functions_yes_no OP functions_yes_no COMP_OP INTEGERS",
                                    "functions_yes_no OP functions_yes_no COMP_OP variable"
        ]
        self._grammar['variable'] = [
                                        "odds",
                                        "evens",
                                        "highs",
                                        "lows",
                                        "marker",
                                        "threshold"
                                        ]
        self._grammar['expression_col'] = [
                                        "INTEGERS",
                                        "variable_col OP INTEGERS",
                                        "variable_col OP variable_col",
                                        "variable_col OP expression_col",
                                        "variable_col OP functions_col",

                                        "expression_col OP INTEGERS",
                                        "expression_col OP variable_col",
                                        "expression_col OP functions_col",

                                        "functions_col OP INTEGERS",
                                        "functions_col OP functions_col"
                                        ]
        self._grammar['variable_col'] = [
                                        "odds",
                                        "evens",
                                        "highs",
                                        "lows",
                                        "marker",
                                        "threshold",
                                        'move_value[column]',
                                        'progress_value[column]',
                                        ]
        self._grammar['vector'] = [
                                    'move_value',
                                    'progress_value'
                                ]
        self._grammar['functions_bool'] = [
                                        'ExperimentalDSL.will_player_win_after_n(state)',
                                        'ExperimentalDSL.are_there_available_columns_to_play(state)'
                                    ]
        self._grammar['functions_yes_no'] = [# Strictly "string" actions
                                'ExperimentalDSL.calculate_score(state, vector )',
                                'ExperimentalDSL.calculate_difficulty_score(state, odds, evens, highs, lows)',
                                'ExperimentalDSL.get_player_total_advance(state)',
                                'ExperimentalDSL.get_opponent_total_advance(state)',
                                'ExperimentalDSL.number_cells_advanced_this_round(state)',
                                ]
        self._grammar['functions_col'] = [# Strictly "numeric" actions
                                'ExperimentalDSL.is_new_neutral(actions[i], state)',
                                'ExperimentalDSL.advance(actions[i])',
                                ]
        self._grammar['OP'] = ['+', '-', '*']
        self._grammar['BOOL_OP'] = ['and', 'or', 'and not', 'or not']
        self._grammar['COMP_OP'] = ['>', '<', '>=', '<=']
        self._grammar['INTEGERS'] = [str(i) for i in range(1, 40)]
        # Used in the parse tree to finish expanding hanging nodes
        self.finishable_nodes = ['header', 'body_if', 'body_else', 'bool_template', 
                                'bool_exp', 'variable', 'expression_col', 'variable_col',
                                'vector', 'functions_bool', 'functions_yes_no', 
                                'functions_col', 'OP', 'BOOL_OP', 'COMP_OP', 'INTEGERS']

        # Dictionary to "quickly" finish the tree.
        # Needed for the tree to not surpass the max node limit.
        self.quickly_finish = {}

    def set_type_action(self, string_action):
        '''
        - string_action is a boolean. True if it is a y/n action, False if it 
          is a column action.
        '''
        if string_action:
            self._grammar[self.start] = [r"\t \t header \n \t \t body_if"]
        else:
            self._grammar[self.start] = [r"\n \t \t body_else"]
            
        self.quickly_finish = {
                                self.start :self._grammar[self.start],
                                'header' :self._grammar['header'],
                                'body_if' :self._grammar['body_if'],
                                'body_else' :self._grammar['body_else'],
                                'bool_template' :["bool_exp", "bool_exp BOOL_OP bool_exp"],
                                'bool_exp' :self._grammar['bool_exp'],
                                'variable' :self._grammar['variable'],
                                'expression_col' :["INTEGERS",
                                                    "variable_col OP INTEGERS",
                                                    "variable_col OP variable_col",
                                                    "variable_col OP functions_col",
                                                    "functions_col OP INTEGERS",
                                                    "functions_col OP functions_col"],
                                'variable_col' :self._grammar['variable_col'],
                                'vector' :self._grammar['vector'],
                                'functions_bool' :self._grammar['functions_bool'],
                                'functions_yes_no' :self._grammar['functions_yes_no'],
                                'functions_col' :self._grammar['functions_col'],
                                'OP' :self._grammar['OP'],
                                'BOOL_OP' :self._grammar['BOOL_OP'],
                                'COMP_OP' :self._grammar['COMP_OP'],
                                'INTEGERS' :self._grammar['INTEGERS']
                            }

    @staticmethod
    def get_player_total_advance(state):
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
    def get_opponent_total_advance(state):
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
    def calculate_score(state, progress_value):
        score = 0
        neutrals = [col[0] for col in state.neutral_positions]
        for col in neutrals:
            advance = ExperimentalDSL.number_cells_advanced_this_round_for_col(state, col)
            # +1 because whenever a neutral marker is used, the weight of that
            # column is summed
            score += (advance + 1) * progress_value[col]
        return score

    @staticmethod
    def number_cells_advanced_this_round(state):
        """
        Return the number of positions advanced in this round for current
        player for all columns.
        """
        counter = 0
        for column in range(state.column_range[0], state.column_range[1]+1):
            counter += ExperimentalDSL.number_cells_advanced_this_round_for_col(state, column)
        return counter

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
    def get_available_columns(state):
        """ Return a list of all available columns. """

        # List containing all columns, remove from it the columns that are
        # available given the current board
        available_columns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for neutral in state.neutral_positions:
            available_columns.remove(neutral[0])
        for finished in state.finished_columns:
            if finished[0] in available_columns:
                available_columns.remove(finished[0])

        return available_columns

    @staticmethod
    def will_player_win_after_n(state):
        """ 
        Return a boolean in regards to if the player will win the game or not 
        if they choose to stop playing the current round (i.e.: choose the 
        'n' action). 
        """
        clone_state = state.clone()
        clone_state.play('n')
        won_columns = 0
        for won_column in clone_state.finished_columns:
            if state.player_turn == won_column[1]:
                won_columns += 1
        #This means if the player stop playing now, they will win the game
        if won_columns == 3:
            return True
        else:
            return False

    @staticmethod
    def are_there_available_columns_to_play(state):
        """
        Return a booleanin regards to if there available columns for the player
        to choose. That is, if the does not yet have all three neutral markers
        used AND there are available columns that are not finished/won yet.
        """
        available_columns = ExperimentalDSL.get_available_columns(state)
        return state.n_neutral_markers != 3 and len(available_columns) > 0

    @staticmethod
    def calculate_difficulty_score(state, odds, evens, highs, lows):
        """
        Add an integer to the current score given the peculiarities of the
        neutral marker positions on the board.
        """
        difficulty_score = 0

        neutral = [n[0] for n in state.neutral_positions]
        # If all neutral markers are in odd columns
        if all([x % 2 != 0 for x in neutral]):
            difficulty_score += odds
        # If all neutral markers are in even columns
        if all([x % 2 == 0 for x in neutral]):
            difficulty_score += evens
        # If all neutral markers are is "low" columns
        if all([x <= 7 for x in neutral]):
            difficulty_score += lows
        # If all neutral markers are is "high" columns
        if all([x >= 7 for x in neutral]):
            difficulty_score += highs

        return difficulty_score

    @staticmethod
    def is_new_neutral(action, state):
        # Return a boolean representing if action will place a new neutral. """
        is_new_neutral = True
        for neutral in state.neutral_positions:
            if neutral[0] == action:
                is_new_neutral = False

        return is_new_neutral

    @staticmethod
    def advance(action):
        """ Return how many cells this action will advance for each column. """

        # Special case: doubled action (e.g. (6,6))
        if len(action) == 2 and action[0] == action[1]:
            return 2
        # All other cases will advance only one cell per column
        else:
            return 1