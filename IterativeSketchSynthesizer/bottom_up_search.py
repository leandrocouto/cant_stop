import random
import sys
from DSL import *
sys.path.insert(0,'..')
from game import Board, Game

class ProgramList:
    
    def __init__(self):

        self.p_list = {}
        self.number_programs = 0
    
    def insert(self, program):

        if program.getSize() not in self.p_list:
            self.p_list[program.getSize()] = {}
        
        if program.className() not in self.p_list[program.getSize()]:
            self.p_list[program.getSize()][program.className()] = []
        
        self.p_list[program.getSize()][program.className()].append(program)
        self.number_programs += 1
        
    def init_plist(self, partialDSL):

        for i in partialDSL.variables_scalar_from_array:
            self.insert(i)
                
        for i in partialDSL.variables_list:
            self.insert(i)

        for i in partialDSL.constant_values:
            self.insert(i)
            
        for i in partialDSL.functions_scalars:
            self.insert(i)
                                                
    def get_programs(self, size):
        
        if size in self.p_list: 
            return self.p_list[size]
        return {}
    
    def get_number_programs(self):
        return self.number_programs

class BottomUpSearch:
    
    def grow(self, partial_DSL, size):
        new_programs = []
        for op in partial_DSL.operations:
            # Special case: root
            if op.className() == 'HoleNode':
                for p in op.grow(self.p_list, size, partial_DSL, True):
                    if p.to_string() not in self.closed_list:
                        self.closed_list.add(p.to_string())
                        self.closed_list_objects.add(p)
                        new_programs.append(p)
                        yield p 
            else:
                for p in op.grow(self.p_list, size, partial_DSL):
                    if p.to_string() not in self.closed_list:
                        self.closed_list.add(p.to_string())
                        self.closed_list_objects.add(p)
                        new_programs.append(p)
                        yield p 
        for p in new_programs:
            self.p_list.insert(p)
            
    def search(self, bound, partial_DSL):     
        
        self.closed_list = set()
        self.closed_list_objects = set()
        
        self.p_list = ProgramList()
        self.p_list.init_plist(partial_DSL)
        current_size = 0

        while current_size <= bound:
            for p in self.grow(partial_DSL, current_size):
                pass
            current_size += 1
        
        return self.closed_list, self.closed_list_objects

    def synthesize(self, bound, partial_DSL, folder):

        n_partial_terms = len(partial_DSL.get_all_terms())

        with open(folder + 'PBUS_' + str(n_partial_terms) + '_DSLTerms.txt', 'a') as f:
            print('Partial DSL used', file=f)
            print(partial_DSL.get_all_terms(), file=f)
            print(file=f)

        closed_list, closed_list_objects = self.search(bound, partial_DSL)

        with open(folder + 'PBUS_' + str(n_partial_terms) + '_DSLTerms.txt', 'a') as f:
            print('Closed list collected - Length = ', len(closed_list),  file=f)
            for p in closed_list:
                print(p, file=f)
            print(file=f)
                      
        return (closed_list, closed_list_objects,)


