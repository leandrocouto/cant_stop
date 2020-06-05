import random

class Script:
    def __init__(self, numeric_rules, string_rules, id=0):
        self._numeric_rules = numeric_rules
        self._string_rules = string_rules
        self._id = id
        self._py = r'''
from players.player import Player
import random

class Script{0}(Player):

    def __init__(self):
        self._counter_calls = []
        for i in range({1}):
            self._counter_calls.append(0)
            
    def get_counter_calls(self):
        return self._counter_calls 

    def get_action(self, state):
        actions = state.available_moves()
        
        for a in actions:
        	if a in ['y', 'n']:
        '''
        
        self._if_string = r'''
            	{0}:
                	self._counter_calls[{1}] += 1
                	return a
                    	'''
        self._else_string = r'''
        	else:
            	{0}:
                	self._counter_calls[{1}] += 1
                	return a
                    	'''
        self._end_script = r'''
        return actions[0]
                    '''        
    
    def _generateTextScript(self):
        
        number_string_rules = len(self._string_rules)
        py = self._py.format(str(self._id), str(number_string_rules))

        for i in range(number_string_rules):
            py += self._if_string.format(self._string_rules[i], i)

        number_numeric_rules = len(self._numeric_rules)
        py = self._py.format(str(self._id), str(number_numeric_rules))

        for i in range(number_numeric_rules):
            py += self._if_string.format(self._rules[i], i)

        py += self._end_script
        
        return py
                        
    def saveFile(self, path):
        py = self._generateTextScript()
        
        file = open(path + 'Script'+ str(self._id) + '.py', 'w')
        file.write(py)
        file.close()