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
        
        print('Number of programs: ', self.plist.get_number_programs())
        
        self._variables_list = variables_list
        self._variables_scalar_from_array = variables_scalar_from_array

        number_evaluations = 0
        current_size = 0

        while current_size <= bound:
            
            number_evaluations_bound = 0

            for p in self.grow(operations, current_size):
                number_evaluations += 1
                number_evaluations_bound += 1
#                 if type(p).__name__ == 'Sum':      
#                     print(p.toString(), p.getSize(), current_size)
#                 correct, error = self.is_correct(p)

                if p.toString() in programs_to_not_eval or type(p).__name__ != 'Argmax':
                    continue

                #print('p.tostring = ', p.toString())
                correct, error = self.eval_function.eval(p)
                #print('correct error = ', correct, error)
#                 if error:
#                     print('Program ', p.toString(), ' raised an exception')
       
                if correct:
                    print('encontrou um correct')
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

# synthesizer = BottomUpSearch()
# 
# p, num = synthesizer.synthesize(38, [Sum, Map, Function, Plus, Sum], [1, 2], ['arg'], [],
#                                 [{'arg':[1, 2, 3], 'out':6}])
# 
# print(p.toString())

# game = Game(n_players = 2, dice_number = 4, dice_value = 6, column_range = [2,12],
#             offset = 2, initial_height = 3)
# 
# for _ in range(50):
#     actions = game.available_moves()
#     game.play(random.choice(actions))
# 
# progress_value = [0, 0, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6]
# 
# env = {}
# env['state'] = game
# env['progress_value'] = progress_value
# env['neutrals'] = [col[0] for col in game.neutral_positions]
# 
# print(env['progress_value'])

# print(p.toString())
# env = {}
# env['neutrals'] = [0, 1, 2]
# env['test'] = [4, 2, 7]
# env['x'] = 1
# s = Sum(Var('neutrals'))
# s = VarList('neutrals')
# s = VarScalar('x')
# s = Plus(VarScalar('x'), VarScalar('x'))
# s = Map(Function(Plus(VarScalar('x'), VarScalarFromArray('test'))), VarList('neutrals'))
# print(s.interpret(env))
# print(s.interpret_local_variables(env,1))

# x = 2
# print(type(x).__name__)
# print(type([1, 2, 3]).__name__)
# print(type((1, 2, 3)).__name__)

# f = lambda x, env : x + 1 if env['test'] = x else None
# print('Printing lambda: ', f(4, env))
# print(env)



