import random
import codecs

class Sketch:
    def __init__(self, program_string, program_column, iterations, tree_max_nodes):
        self.program_string = program_string
        self.program_column = program_column
        self.iterations = iterations
        self.tree_max_nodes = tree_max_nodes
        self._py = r'''
from players.player import Player


class Script_{0}(Player):
\tdef get_action(self, state):
\t\timport numpy as np
\t\tactions = state.available_moves()
'''
        
        self.program = r'''{0}'''
    def _generateTextScript(self, file_name):
        py = self._py.format(file_name)
        py += self.program.format(self.program_string)
        py += self.program.format('\n')
        py += self.program.format(self.program_column)
        py = codecs.decode(py, 'unicode_escape')
        return py

    def save_file_custom(self, path, file_name):
        py = self._generateTextScript(file_name)
        with open(path + 'Script_'+ file_name + '.py', 'w') as f:
            f.write(py)
