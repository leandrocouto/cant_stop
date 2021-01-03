class ToyDSL:
    """
    Implementation of a Domain Specific Language (DSL) for the Can't Stop
    domain.
    """
    def __init__(self):
        
        self.start = 'S'
        
        self._grammar = {}
       	
       	self._grammar[self.start] = [r"\n\t\t variable = math_expr \n\t\t return actions[ variable ]"]


        self._grammar['math_expr'] = [
                                        "INTEGERS",
                                        "variable",
                                        "functions_num",
                                        ]


        self._grammar['variable'] = [
                                    "a",
                                    #"b",
                                    #"c"
                                    ]

        self._grammar['functions_num'] = [
                                'DSL.calculate_score(state, vector )',
                                'DSL.calculate_difficulty_score(state, INTEGERS , INTEGERS , INTEGERS , INTEGERS )',
                                ]

        self._grammar['INTEGERS'] = [str(i) for i in range(1, 5)] # 10 = 40k states, # 5 = 2500 states 

        # Used in the parse tree to finish expanding hanging nodes
        self.finishable_nodes = [self.start, 'statement', 'statement_1','statement_2', 'assign_expr', 'math_expr', 'math_term', 
                                'if_expr', 'if_expr_1', 'bool_template', 'bool_expr', 'for_expr', 'for_expr_1',
                                'iterable_for_variable', 'foreach_variable', 'variable', 
                                'vector', 'ret_expr', 'functions_bool', 'functions_num', 
                                'OP', 'BOOL_OP', 'COMP_OP', 'INTEGERS']

        # Dictionary to "quickly" finish the tree.
        # Needed for the tree to not surpass the max node limit.
        self.quickly_finish = {
                                self.start :self._grammar[self.start],
                                'math_expr' : ["math_term"],
                                'variable' :self._grammar['variable'],
                                'functions_num' :self._grammar['functions_num'],
                                'INTEGERS' :self._grammar['INTEGERS'],
                            }