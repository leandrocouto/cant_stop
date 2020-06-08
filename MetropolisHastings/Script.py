import random
import codecs

class Script:
    def __init__(self, program, k, iterations):
        self.program = program
        self.k = k
        self.iterations = iterations
        self.prefix = str(self.k) + "_" + str(self.iterations)
        self._py = r'''
from players.player import Player
import random

class Script_{0}(Player):
\tdef get_action(self, state):
\t\tactions = state.available_moves()
'''
        
        self.program_string = r'''{0}'''
        self._end_script = r'''\n\t\treturn actions[0]'''        
    
    def _generateTextScript(self):
        py = self._py.format(self.prefix)
        py += self.program_string.format(self.program)  
        py += self._end_script
        py = codecs.decode(py, 'unicode_escape')
        return py
                        
    def saveFile(self, path):
        py = self._generateTextScript()
        
        file = open(path + 'Script_'+ str(self.prefix) + '.py', 'w')
        file.write(py)
        file.close()