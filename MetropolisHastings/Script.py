import random
import codecs

class Script:
    def __init__(self, program, k, iterations, tree_max_nodes):
        self.program = program
        self.k = k
        self.iterations = iterations
        self.tree_max_nodes = tree_max_nodes
        self.prefix = str(self.k) + "d_" + str(self.iterations) + "i_" \
                        + str(self.tree_max_nodes) + "n"
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
    def _generateTextScript(self):
        py = self._py.format(self.prefix)
        py += self.program_string.format(self.program)
        py = codecs.decode(py, 'unicode_escape')
        return py
                        
    def saveFile(self, path):
        py = self._generateTextScript()
        
        file = open(path + 'Script_'+ str(self.prefix) + '.py', 'w')
        file.write(py)
        file.close()

    def save_file_custom(self, path, file_name):
        py = self._generateTextScript()
        
        file = open(path + 'Script_'+ file_name + '.py', 'w')
        file.write(py)
        file.close()
