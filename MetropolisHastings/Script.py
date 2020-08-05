import random
import codecs

class Script:
    def __init__(self, program, iterations, tree_max_nodes):
        self.program = program
        self.iterations = iterations
        self.tree_max_nodes = tree_max_nodes
        self._py = r'''
from players.player import Player


class Script_{0}(Player):
\tdef get_action(self, state):
\t\timport numpy as np
\t\tactions = state.available_moves()
\t\tscore_yes_no = 0
\t\tscore_columns = np.zeros(len(actions))
'''
        
        self.program_string = r'''{0}'''
    def _generateTextScript(self, file_name):
        py = self._py.format(file_name)
        py += self.program_string.format(self.program)
        py = codecs.decode(py, 'unicode_escape')
        return py

    def save_file_custom(self, path, file_name):
        py = self._generateTextScript(file_name)
        
        file = open(path + 'Script_'+ file_name + '.py', 'w')
        file.write(py)
        file.close()
