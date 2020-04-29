import random

class Script:
    def __init__(self, rules, id=0):
        self._rules = rules
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
        '''
        
        self._if_string = r'''
            {0}:
                self._counter_calls[{1}] += 1
                return a, 0
                    '''
        self._end_script = r'''
        return actions[0], 1
                    '''        
    
    def _generateTextScript(self):
        
        number_rules = len(self._rules)
        py = self._py.format(str(self._id), str(number_rules))

        for i in range(number_rules):
            py += self._if_string.format(self._rules[i], i)

        py += self._end_script
        
        return py
                        
    def saveFile(self, path):
        py = self._generateTextScript()
        
        file = open(path + 'Script'+ str(self._id) + '.py', 'w')
        file.write(py)
        file.close()