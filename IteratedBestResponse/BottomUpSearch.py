import random
import sys
from DSL import Sum, VarList, Function, VarScalar, VarScalarFromArray, Plus, Map, Constant
sys.path.insert(0,'..')
from game import Board, Game

class ProgramList:
    
    def __init__(self):

        self.plist = {}
        self.number_programs = 0
    
    def insert(self, program):

        if program.getSize() not in self.plist:
            self.plist[program.getSize()] = {}
        
        if program.className() not in self.plist[program.getSize()]:
            self.plist[program.getSize()][program.className()] = []
        
        self.plist[program.getSize()][program.className()].append(program)
        self.number_programs += 1
        
    def init_plist(self, constant_values, variables_list, variables_scalar_from_array, 
        functions_scalars):

        for i in variables_scalar_from_array:
            p = VarScalarFromArray(i)
            self.insert(p)
                
        for i in variables_list:
            p = VarList(i)
            self.insert(p)

        for i in constant_values:
            constant = Constant(i)
            self.insert(constant)
            
        for i in functions_scalars:
            p = i()
            self.insert(p)
                                                
    def get_programs(self, size):
        
        if size in self.plist: 
            return self.plist[size]
        return {}
    
    def get_number_programs(self):
        return self.number_programs

class BottomUpSearch:
    
    def grow(self, operations, size):
        new_programs = []
        for op in operations:
            for p in op.grow(self.plist, size):
                if p.toString() not in self.closed_list:
                    self.closed_list.add(p.toString())
                    new_programs.append(p)
                    yield p 
        for p in new_programs:
            self.plist.insert(p)

            
    def get_closed_list(self):

        return self.closed_list
            
    def search(self, bound, operations, constant_values, variables_list, 
        variables_scalar_from_array, functions_scalars, programs_to_not_eval):     
        
        self.closed_list = set()
        self._outputs = set()
        
        self.plist = ProgramList()
        self.plist.init_plist(constant_values, variables_list, variables_scalar_from_array, functions_scalars)
        
        self._variables_list = variables_list
        self._variables_scalar_from_array = variables_scalar_from_array

        number_evaluations = 0
        current_size = 0

        while current_size <= bound:
            
            number_evaluations_bound = 0

            for p in self.grow(operations, current_size):
                number_evaluations += 1
                number_evaluations_bound += 1

                if p.toString() in programs_to_not_eval or type(p).__name__ != 'Argmax':
                    continue
                correct, error = self.eval_function.eval(p)
       
                if correct:
                    return True, p, number_evaluations
                
            print('Size: ', current_size, ' Evaluations: ', number_evaluations_bound)
            current_size += 1
        
        return True, None, number_evaluations

    def synthesize(self, bound, operations, constant_values, variables_list, 
                   variables_scalar_from_array, functions_scalars, eval_function,
                   programs_to_not_eval):

        has_finished = False
        
        self.eval_function = eval_function
        while not has_finished:
            has_finished, solution, evals = self.search(
                                                    bound, 
                                                    operations, 
                                                    constant_values,  
                                                    variables_list, 
                                                    variables_scalar_from_array,
                                                    functions_scalars,
                                                    programs_to_not_eval
                                                )
                        
        return solution, evals


