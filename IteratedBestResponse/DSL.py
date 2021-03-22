import numpy as np
import itertools

class Node:
    def __init__(self):
        self.size = 0
        self.local = 'locals'
        self.intname = 'int'
        self.listname = 'list'
        self.tuplename = 'tuple'
        self.statename = 'state'
    
    def getSize(self):
        return self.size
    
    def toString(self):
        raise Exception('Unimplemented method: toString')
    
    def interpret(self):
        raise Exception('Unimplemented method: interpret')
    
    def interpret_local_variables(self, env, x):        
        if self.local not in env:
            env[self.local] = {}
        
        if type(x).__name__ == self.tuplename:
            x = list(x)
        
        env[self.local][type(x).__name__] = x
                
        return self.interpret(env) 
    
    def getRulesNames(self, rules):
        raise Exception('Unimplemented method: getRulesNames')
    
    @classmethod
    def grow(plist, new_plist):
        pass
    
    @classmethod
    def className(cls):
        return cls.__name__

class VarList(Node):
    def __init__(self, name):
        self.name = name
        self.size = 1
        
    def toString(self):
        return self.name
    
    def interpret(self, env):
        return env[self.name]
    
class VarScalarFromArray(Node):
    def __init__(self, name):
        super(VarScalarFromArray, self).__init__()
        self.name = name
        self.size = 1
        
    def toString(self):
        return self.name
    
    def interpret(self, env):
        return env[self.name][env[self.local][self.intname]]
    
class VarScalar(Node):
    def __init__(self, name):
        self.name = name
        self.size = 1
        
    def toString(self):
        return self.name
    
    def interpret(self, env):
        return env[self.name]
    
class Constant(Node):
    def __init__(self, value):
        self.value = value
        self.size = 1
        
    def toString(self):
        return str(self.value)
    
    def interpret(self, env):
        return self.value

class NumberAdvancedByAction(Node):
    def __init__(self):
        super(NumberAdvancedByAction, self).__init__()
        self.size = 1
        
    def toString(self):
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

class IsNewNeutral(Node):
    def __init__(self):
        super(IsNewNeutral, self).__init__()
        self.size = 1
        
    def toString(self):
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


class NumberAdvancedThisRound(Node):
    def __init__(self):
        super(NumberAdvancedThisRound, self).__init__()
        self.size = 1
        
    def toString(self):
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

class Times(Node):
    def __init__(self, left, right):
        super(Times, self).__init__()
        self.left = left
        self.right = right
        self.size = self.left.size + self.right.size + 1
        
    def toString(self):
        return "(" + self.left.toString() + " * " + self.right.toString() + ")"
    
    def interpret(self, env):
        return self.left.interpret(env) * self.right.interpret(env)
    
    def grow(plist, size):       
        new_programs = []
        # defines which nodes are accepted in the AST
#         accepted_nodes = set([VarScalar.className(), 
#                               VarScalarFromArray.className(), 
#                               NumberAdvancedThisRound.className(),
#                               Constant.className(),
#                               Plus.className(),
#                               Times.className(),
#                               Minus.className()])
        accepted_nodes = set([VarScalar.className(), 
                              VarScalarFromArray.className(), 
                              NumberAdvancedThisRound.className(),
                              NumberAdvancedByAction.className(),
                              Constant.className()])
        
        # generates all combinations of cost of size 2 varying from 1 to size - 1
        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for c in combinations:           
            # skip if the cost combination exceeds the limit
            if c[0] + c[1] + 1 != size:
                continue
                    
            # retrive bank of programs with costs c[0], c[1], and c[2]
            program_set1 = plist.get_programs(c[0])
            program_set2 = plist.get_programs(c[1])
                
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
        
    def toString(self):
        return "(" + self.left.toString() + " - " + self.right.toString() + ")"
    
    def interpret(self, env):
        return self.left.interpret(env) - self.right.interpret(env)
    
    def grow(plist, size):               
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([VarScalar.className(), 
                      VarScalarFromArray.className(), 
                      NumberAdvancedThisRound.className(),
                      NumberAdvancedByAction.className(),
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
            program_set1 = plist.get_programs(c[0])
            program_set2 = plist.get_programs(c[1])
                
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
        
    def toString(self):
        return "(" + self.left.toString() + " + " + self.right.toString() + ")"
    
    def interpret(self, env):
        return self.left.interpret(env) + self.right.interpret(env)
    
    def grow(plist, size):               
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([VarScalar.className(), 
                      VarScalarFromArray.className(), 
                      NumberAdvancedThisRound.className(),
                      NumberAdvancedByAction.className(),
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
            program_set1 = plist.get_programs(c[0])
            program_set2 = plist.get_programs(c[1])
                
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
    
                            plus = Plus(p1, p2)
                            new_programs.append(plus)
            
                            yield plus
        return new_programs  
    

class Function(Node):
    def __init__(self, expression):
        super(Function, self).__init__()
        self.expression = expression
        self.size = self.expression.size + 1
        
    def toString(self):
        return "(lambda x : " + self.expression.toString() + ")"
    
    def interpret(self, env):
        return lambda x : self.expression.interpret_local_variables(env, x) 
    
    def grow(plist, size):
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([Minus.className(), Plus.className(), Times.className(), Sum.className(), Map.className(), Function.className()])
         
        program_set = plist.get_programs(size - 1)
                    
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
        
    def toString(self):
        return 'argmax(' + self.list.toString() + ")"
    
    def interpret(self, env):
        return np.argmax(self.list.interpret(env)) 
    
    def grow(plist, size):       
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([VarList.className(), Map.className()])
        program_set = plist.get_programs(size - 1)
                    
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
        
    def toString(self):
        return 'sum(' + self.list.toString() + ")"
    
    def interpret(self, env):
        return np.sum(self.list.interpret(env)) 
    
    def grow(plist, size):       
        new_programs = []
        # defines which nodes are accepted in the AST
        accepted_nodes = set([VarList.className(), Map.className()])
        program_set = plist.get_programs(size - 1)
                    
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
        
        if self.list is None:
            self.size = self.function.size + 1
        else:
            self.size = self.list.size + self.function.size + 1
        
    def toString(self):
        if self.list is None:
            return 'map(' + self.function.toString() + ", None)"
        
        return 'map(' + self.function.toString() + ", " + self.list.toString() + ")"
    
    def interpret(self, env):
        # if list is None, then it tries to retrieve from local variables from a lambda function
        if self.list is None:
            list_var = env[self.local][self.listname]
            return list(map(self.function.interpret(env), list_var))
        
        return list(map(self.function.interpret(env), self.list.interpret(env))) 
    
    def grow(plist, size):  
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
                    
            # retrive bank of programs with costs c[0], c[1], and c[2]
            program_set1 = plist.get_programs(c[0])
            program_set2 = plist.get_programs(c[1])
            
            if c[1] == 0:
                if VarList.className() not in program_set2:
                    program_set2[VarList.className()] = []
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
    
                            m = Map(p1, p2)
                            new_programs.append(m)
            
                            yield m
        return new_programs 