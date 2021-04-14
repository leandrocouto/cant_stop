import numpy as np
import random
import itertools
import pickle

class Node:
    def __init__(self):
        self.size = 0
        self.local = 'locals'
        self.intname = 'int'
        self.listname = 'list'
        self.tuplename = 'tuple'
        self.statename = 'state'
        self.parent = None
        self.children = []
        self.id = 0
        self.can_mutate = True
    
    def getSize(self):
        return self.size
    
    def to_string(self):
        raise Exception('Unimplemented method: to_string')
    
    def interpret(self):
        raise Exception('Unimplemented method: interpret')
    
    def interpret_local_variables(self, env, x):        
        if self.local not in env:
            env[self.local] = {}
        
        if type(x).__name__ == self.tuplename:
            x = list(x)
        
        env[self.local][type(x).__name__] = x
                
        return self.interpret(env) 

    def print_tree(self):
        indentation = '  '
        self._print_tree(self, indentation)

    def _print_tree(self, node, indentation):
        #For root
        if indentation == '  ':
            print(node, ' - ', node.to_string(), ' - ', node.id, ' - can mutate = ', node.can_mutate)
        else:
            print(indentation, node, ' - ', node.to_string(), ' - ', node.id, ' - can mutate = ', node.can_mutate)
        if hasattr(node, 'children'):
            for child in node.children:
                self._print_tree(child, indentation + '    ')

    def getRulesNames(self, rules):
        raise Exception('Unimplemented method: getRulesNames')

    def add_parent(self, parent):
        raise Exception('Unimplemented method: add_parent - class:', self, 'parent:', parent)

    def add_children(self, children):
        raise Exception('Unimplemented method: add_children - class:', self, 'children:', children)
    
    @classmethod
    def grow(p_list, new_p_list):
        pass
    
    @classmethod
    def className(cls):
        return cls.__name__

class HoleNode(Node):
    def __init__(self):
        super(HoleNode, self).__init__()
        self.size = 1
        self.can_mutate = True
        
    def to_string(self):
        return '?'
    
    def interpret(self, env):
        return

    def add_parent(self, parent):
        self.parent = parent

class NoneNode(Node):
    def __init__(self):
        super(NoneNode, self).__init__()
        self.size = 1
        
    def to_string(self):
        return 'None'
    
    def interpret(self, env):
        return

    def add_parent(self, parent):
        self.parent = parent

class VarList(Node):
    def __init__(self, name):
        super(VarList, self).__init__()
        self.name = name
        self.size = 1
        
    def to_string(self):
        return self.name
    
    def interpret(self, env):
        return env[self.name]

    def add_parent(self, parent):
        self.parent = parent
    
class VarScalarFromArray(Node):
    def __init__(self, name):
        super(VarScalarFromArray, self).__init__()
        self.name = name
        self.size = 1
        
    def to_string(self):
        return self.name
    
    def interpret(self, env):
        return env[self.name][env[self.local][self.intname]]

    def add_parent(self, parent):
        self.parent = parent
    
class VarScalar(Node):
    def __init__(self, name):
        super(VarScalar, self).__init__()
        self.name = name
        self.size = 1
        
    def to_string(self):
        return self.name
    
    def interpret(self, env):
        return env[self.name]

    def add_parent(self, parent):
        self.parent = parent
    
class Constant(Node):
    def __init__(self, value):
        self.value = value
        self.size = 1
        
    def to_string(self):
        return str(self.value)
    
    def interpret(self, env):
        return self.value

    def add_parent(self, parent):
        self.parent = parent

class NumberAdvancedByAction(Node):
    def __init__(self):
        super(NumberAdvancedByAction, self).__init__()
        self.size = 1
        
    def to_string(self):
        return type(self).__name__
    
    def interpret(self, env):
        """
        Return the number of positions advanced in this round for a given
        column by the player.
        """
        action = env[self.local][self.listname]
        
        # Special case: doubled action (e.g. (6,6))
        if len(action) == 2 and action[0] == action[1]:
            return 2
        # All other cases will advance only one cell per column
        else:
            return 1

    def add_parent(self, parent):
        self.parent = parent

class IsNewNeutral(Node):
    def __init__(self):
        super(IsNewNeutral, self).__init__()
        self.size = 1
        
    def to_string(self):
        return type(self).__name__
    
    def interpret(self, env):
        """
        Return the number of positions advanced in this round for a given
        column by the player.
        """
        state = env[self.statename]
        column = env[self.local][self.intname]
        
        # Return a boolean representing if action will place a new neutral. """
        is_new_neutral = True
        for neutral in state.neutral_positions:
            if neutral[0] == column:
                is_new_neutral = False

        return is_new_neutral

    def add_parent(self, parent):
        self.parent = parent

class NumberAdvancedThisRound(Node):
    def __init__(self):
        super(NumberAdvancedThisRound, self).__init__()
        self.size = 1
        
    def to_string(self):
        return type(self).__name__
    
    def interpret(self, env):
        """
        Return the number of positions advanced in this round for a given
        column by the player.
        """
        state = env[self.statename]
        column = env[self.local][self.intname]
        
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

    def add_parent(self, parent):
        self.parent = parent

class PlayerColumnAdvance(Node):
    def __init__(self):
        super(PlayerColumnAdvance, self).__init__()
        self.size = 1
        
    def to_string(self):
        return type(self).__name__
    
    def interpret(self, env):
        """ 
        Get the number of cells advanced in a specific column by the player. 
        """

        state = env[self.statename]
        column = env[self.local][self.intname]

        if column not in list(range(2,13)):
            raise Exception('Out of range column passed to PlayerColumnAdvance()')
        counter = 0
        player = state.player_turn
        # First check if the column is already won
        for won_column in state.finished_columns:
            if won_column[0] == column and won_column[1] == player:
                return len(state.board_game.board[won_column[0]]) + 1
            elif won_column[0] == column and won_column[1] != player:
                return 0
        # If not, 'manually' count it while taking note of the neutral position
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

    def add_parent(self, parent):
        self.parent = parent

class OpponentColumnAdvance(Node):
    def __init__(self):
        super(OpponentColumnAdvance, self).__init__()
        self.size = 1
        
    def to_string(self):
        return type(self).__name__
    
    def interpret(self, env):
        """ 
        Get the number of cells advanced in a specific column by the opponent. 
        """

        state = env[self.statename]
        column = env[self.local][self.intname]

        if column not in list(range(2,13)):
            raise Exception('Out of range column passed to OpponentColumnAdvance()')
        counter = 0
        if state.player_turn == 1:
            player = 2
        else:
            player = 1
        # First check if the column is already won
        for won_column in state.finished_columns:
            if won_column[0] == column and won_column[1] == player:
                return len(state.board_game.board[won_column[0]]) + 1
            elif won_column[0] == column and won_column[1] != player:
                return 0
        # If not, 'manually' count it
        previously_conquered = -1
        neutral_position = -1
        list_of_cells = state.board_game.board[column]

        for i in range(len(list_of_cells)):
            if player in list_of_cells[i].markers:
                previously_conquered = i

        return previously_conquered + 1

    def add_parent(self, parent):
        self.parent = parent

class Times(Node):
    def __init__(self, left, right):
        super(Times, self).__init__()
        self.left = left
        self.right = right
        self.size = self.left.size + self.right.size + 1
        
    def to_string(self):
        return "(" + self.left.to_string() + " * " + self.right.to_string() + ")"
    
    def interpret(self, env):
        return self.left.interpret(env) * self.right.interpret(env)

    def add_parent(self, parent):
        self.parent = parent
        # In case there's already data in there
        self.children = []
        self.children.append(self.left)
        self.children.append(self.right)

    def add_children(self, children):
        self.left = children[0]
        self.right = children[1]
    
    def grow(p_list, size):       
        new_programs = []
        accepted_nodes = set([VarScalar.className(), 
                              VarScalarFromArray.className(), 
                              NumberAdvancedThisRound.className(),
                              NumberAdvancedByAction.className(),
                              IsNewNeutral.className(),
                              PlayerColumnAdvance.className(),
                              OpponentColumnAdvance.className(),
                              Constant.className()])
        
        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:           
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue
                    
            # retrive bank of programs with costs c[0], c[1], and c[2]
            program_set1 = p_list.get_programs(c[0])
            program_set2 = p_list.get_programs(c[1])
                
            for t1, programs1 in program_set1.items():                
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes:
                    continue
                
                for p1 in programs1:                       

                    for t2, programs2 in program_set2.items():                
                        # skip if t1 isn't a node accepted by Lt
                        if t2 not in accepted_nodes:
                            continue
                        
                        for p2 in programs2:
                            if isinstance(p1, NumberAdvancedThisRound):
                                p1 = NumberAdvancedThisRound()
                            elif isinstance(p1, NumberAdvancedByAction):
                                p1 = NumberAdvancedByAction()
                            elif isinstance(p1, IsNewNeutral):
                                p1 = IsNewNeutral()
                            elif isinstance(p1, PlayerColumnAdvance):
                                p1 = PlayerColumnAdvance()
                            elif isinstance(p1, OpponentColumnAdvance):
                                p1 = OpponentColumnAdvance()
                            elif isinstance(p1, VarScalarFromArray):
                                p1 = VarScalarFromArray(p1.name)
                            elif isinstance(p1, VarScalar):
                                p1 = VarScalar(p1.name)

                            if isinstance(p2, NumberAdvancedThisRound):
                                p2 = NumberAdvancedThisRound()
                            elif isinstance(p2, NumberAdvancedByAction):
                                p2 = NumberAdvancedByAction()
                            elif isinstance(p2, IsNewNeutral):
                                p2 = IsNewNeutral()
                            elif isinstance(p2, PlayerColumnAdvance):
                                p2 = PlayerColumnAdvance()
                            elif isinstance(p2, OpponentColumnAdvance):
                                p2 = OpponentColumnAdvance()
                            elif isinstance(p2, VarScalarFromArray):
                                p2 = VarScalarFromArray(p2.name)
                            elif isinstance(p2, VarScalar):
                                p2 = VarScalar(p2.name)

                            times = Times(p1, p2)
                            new_programs.append(times)
            
                            yield times
        return new_programs 

class Minus(Node):
    def __init__(self, left, right):
        super(Minus, self).__init__()
        self.left = left
        self.right = right
        self.size = self.left.size + self.right.size + 1
        
    def to_string(self):
        return "(" + self.left.to_string() + " - " + self.right.to_string() + ")"
    
    def interpret(self, env):
        return self.left.interpret(env) - self.right.interpret(env)

    def add_parent(self, parent):
        self.parent = parent
        # In case there's already data in there
        self.children = []
        self.children.append(self.left)
        self.children.append(self.right)

    def add_children(self, children):
        self.left = children[0]
        self.right = children[1]
    
    def grow(p_list, size):               
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([VarScalar.className(), 
                      VarScalarFromArray.className(), 
                      NumberAdvancedThisRound.className(),
                      NumberAdvancedByAction.className(),
                      IsNewNeutral.className(),
                      PlayerColumnAdvance.className(),
                      OpponentColumnAdvance.className(),
                      Constant.className(),
                      Plus.className(),
                      Times.className(),
                      Minus.className()])
        
        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:                       
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue
                
            # retrive bank of programs with costs c[0], c[1], and c[2]
            program_set1 = p_list.get_programs(c[0])
            program_set2 = p_list.get_programs(c[1])
                
            for t1, programs1 in program_set1.items():                
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes:
                    continue
                
                for p1 in programs1:                       

                    for t2, programs2 in program_set2.items():                
                        # skip if t1 isn't a node accepted by Lt
                        if t2 not in accepted_nodes:
                            continue
                        
                        for p2 in programs2:
                            if isinstance(p1, NumberAdvancedThisRound):
                                p1 = NumberAdvancedThisRound()
                            elif isinstance(p1, NumberAdvancedByAction):
                                p1 = NumberAdvancedByAction()
                            elif isinstance(p1, IsNewNeutral):
                                p1 = IsNewNeutral()
                            elif isinstance(p1, PlayerColumnAdvance):
                                p1 = PlayerColumnAdvance()
                            elif isinstance(p1, OpponentColumnAdvance):
                                p1 = OpponentColumnAdvance()
                            elif isinstance(p1, VarScalarFromArray):
                                p1 = VarScalarFromArray(p1.name)
                            elif isinstance(p1, VarScalar):
                                p1 = VarScalar(p1.name)

                            if isinstance(p2, NumberAdvancedThisRound):
                                p2 = NumberAdvancedThisRound()
                            elif isinstance(p2, NumberAdvancedByAction):
                                p2 = NumberAdvancedByAction()
                            elif isinstance(p2, IsNewNeutral):
                                p2 = IsNewNeutral()
                            elif isinstance(p2, PlayerColumnAdvance):
                                p2 = PlayerColumnAdvance()
                            elif isinstance(p2, OpponentColumnAdvance):
                                p2 = OpponentColumnAdvance()
                            elif isinstance(p2, VarScalarFromArray):
                                p2 = VarScalarFromArray(p2.name)
                            elif isinstance(p2, VarScalar):
                                p2 = VarScalar(p2.name)

                            minus = Minus(p1, p2)
                            new_programs.append(minus)
            
                            yield minus
        return new_programs  

class Plus(Node):
    def __init__(self, left, right):
        super(Plus, self).__init__()
        self.left = left
        self.right = right
        self.size = self.left.size + self.right.size + 1
        
    def to_string(self):
        return "(" + self.left.to_string() + " + " + self.right.to_string() + ")"
    
    def interpret(self, env):
        return self.left.interpret(env) + self.right.interpret(env)

    def add_parent(self, parent):
        self.parent = parent
        # In case there's already data in there
        self.children = []
        self.children.append(self.left)
        self.children.append(self.right)

    def add_children(self, children):
        self.left = children[0]
        self.right = children[1]
    
    def grow(p_list, size):               
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([VarScalar.className(), 
                      VarScalarFromArray.className(), 
                      NumberAdvancedThisRound.className(),
                      NumberAdvancedByAction.className(),
                      IsNewNeutral.className(),
                      PlayerColumnAdvance.className(),
                      OpponentColumnAdvance.className(),
                      Constant.className(),
                      Plus.className(),
                      Times.className(),
                      Minus.className()])
        
        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:                       
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue
                
            # retrieve bank of programs with costs c[0], c[1], and c[2]
            program_set1 = p_list.get_programs(c[0])
            program_set2 = p_list.get_programs(c[1])
                
            for t1, programs1 in program_set1.items():                
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes:
                    continue
                
                for p1 in programs1:                       

                    for t2, programs2 in program_set2.items():            
                        # skip if t1 isn't a node accepted by Lt
                        if t2 not in accepted_nodes:
                            continue
                        
                        for p2 in programs2:
                            if isinstance(p1, NumberAdvancedThisRound):
                                p1 = NumberAdvancedThisRound()
                            elif isinstance(p1, NumberAdvancedByAction):
                                p1 = NumberAdvancedByAction()
                            elif isinstance(p1, IsNewNeutral):
                                p1 = IsNewNeutral()
                            elif isinstance(p1, PlayerColumnAdvance):
                                p1 = PlayerColumnAdvance()
                            elif isinstance(p1, OpponentColumnAdvance):
                                p1 = OpponentColumnAdvance()
                            elif isinstance(p1, VarScalarFromArray):
                                p1 = VarScalarFromArray(p1.name)
                            elif isinstance(p1, VarScalar):
                                p1 = VarScalar(p1.name)

                            if isinstance(p2, NumberAdvancedThisRound):
                                p2 = NumberAdvancedThisRound()
                            elif isinstance(p2, NumberAdvancedByAction):
                                p2 = NumberAdvancedByAction()
                            elif isinstance(p2, IsNewNeutral):
                                p2 = IsNewNeutral()
                            elif isinstance(p2, PlayerColumnAdvance):
                                p2 = PlayerColumnAdvance()
                            elif isinstance(p2, OpponentColumnAdvance):
                                p2 = OpponentColumnAdvance()
                            elif isinstance(p2, VarScalarFromArray):
                                p2 = VarScalarFromArray(p2.name)
                            elif isinstance(p2, VarScalar):
                                p2 = VarScalar(p2.name)

                            plus = Plus(p1, p2)
                            new_programs.append(plus)
            
                            yield plus
        return new_programs
    
class Function(Node):
    def __init__(self, expression):
        super(Function, self).__init__()
        self.expression = expression
        self.size = self.expression.size + 1
        
    def to_string(self):
        return "(lambda x : " + self.expression.to_string() + ")"
    
    def interpret(self, env):
        return lambda x : self.expression.interpret_local_variables(env, x)

    def add_parent(self, parent):
        self.parent = parent
        # In case there's already data in there
        self.children = []
        self.children.append(self.expression)

    def add_children(self, children):
        self.expression = children[0]
    
    def grow(p_list, size):
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Minus.className(), Plus.className(), Times.className(), Sum.className(), Map.className(), Function.className()])
         
        program_set = p_list.get_programs(size - 1)
                    
        for t1, programs1 in program_set.items():                
            # skip if t1 isn't a node accepted by Lt
            if t1 not in accepted_nodes:
                continue
            
            for p1 in programs1:                       
                func = Function(p1)
                new_programs.append(func)
        
                yield func
        return new_programs 

class Argmax(Node):
    def __init__(self, l):
        super(Argmax, self).__init__()
        self.list = l
        self.size = self.list.size + 1
        
    def to_string(self):
        return 'argmax(' + self.list.to_string() + ")"
    
    def interpret(self, env):
        return np.argmax(self.list.interpret(env))

    def add_parent(self, parent):
        self.parent = parent
        # In case there's already data in there
        self.children = []
        self.children.append(self.list)

    def add_children(self, children):
        self.list = children[0]
    
    def grow(p_list, size):       
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([VarList.className(), Map.className()])
        program_set = p_list.get_programs(size - 1)
                    
        for t1, programs1 in program_set.items():                
            # skip if t1 isn't a node accepted by Lt
            if t1 not in accepted_nodes:
                continue
            for p1 in programs1:
                am = Argmax(p1)
                new_programs.append(am)
                yield am
        return new_programs

class Sum(Node):
    def __init__(self, l):
        super(Sum, self).__init__()
        self.list = l
        self.size = self.list.size + 1
        
    def to_string(self):
        return 'sum(' + self.list.to_string() + ")"
    
    def interpret(self, env):
        return np.sum(self.list.interpret(env))

    def add_parent(self, parent):
        self.parent = parent
        # In case there's already data in there
        self.children = []
        self.children.append(self.list)

    def add_children(self, children):
        self.list = children[0]
    
    def grow(p_list, size):       
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([VarList.className(), Map.className()])
        program_set = p_list.get_programs(size - 1)
                    
        for t1, programs1 in program_set.items():             
            # skip if t1 isn't a node accepted by Lt
            if t1 not in accepted_nodes:
                continue
            
            for p1 in programs1:                       

                sum_p = Sum(p1)
                new_programs.append(sum_p)      
                yield sum_p
        return new_programs

class Map(Node):
    def __init__(self, function, l):
        super(Map, self).__init__()
        self.function = function
        self.list = l
        if isinstance(self.list, NoneNode) or self.list is None:
            self.size = self.function.size + 1
        else:
            self.size = self.function.size + self.list.size + 1
        
    def to_string(self):
        return 'map(' + self.function.to_string() + ", " + self.list.to_string() + ")"
    
    def interpret(self, env):
        # if list is None, then it tries to retrieve from local variables from a lambda function
        #if self.list is None:
        if isinstance(self.list, NoneNode):
            list_var = env[self.local][self.listname]
            return list(map(self.function.interpret(env), list_var))
        return list(map(self.function.interpret(env), self.list.interpret(env))) 

    def add_parent(self, parent):
        self.parent = parent
        # In case there's already data in there
        self.children = []
        self.children.append(self.function)
        self.children.append(self.list)

    def add_children(self, children):
        self.function = children[0]
        self.list = children[1]
    
    def grow(p_list, size):  
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes_function = set([Function.className()])
        accepted_nodes_list = set([VarList.className()])
                
        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(0, size), repeat=2))
        for c in combinations:         
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue
                    
            # retrieve bank of programs with costs c[0], c[1], and c[2]
            program_set1 = p_list.get_programs(c[0])
            program_set2 = p_list.get_programs(c[1])
            if c[1] == 0:
                if VarList.className() not in program_set2:
                    program_set2[VarList.className()] = []
                #program_set2[VarList.className()].append(NoneNode())
                program_set2[VarList.className()].append(None)
                
            for t1, programs1 in program_set1.items():                
                # skip if t1 isn't a node accepted by Lt
                if t1 not in accepted_nodes_function:
                    continue
                
                for p1 in programs1:                       
    
                    for t2, programs2 in program_set2.items():                
                        # skip if t1 isn't a node accepted by Lt
                        if t2 not in accepted_nodes_list:
                            continue
                        
                        for p2 in programs2:
                            
                            if p2 is None:
                                p2 = NoneNode()
                            m = Map(p1, p2)
                            new_programs.append(m)
                            yield m
        return new_programs 