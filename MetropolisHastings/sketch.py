import random
import codecs

class Sketch:
    def __init__(self, program_string, program_column, iterations, tree_max_nodes, path):
        self.program_string = program_string
        self.program_column = program_column
        self.iterations = iterations
        self.tree_max_nodes = tree_max_nodes
        self.path = path
        self._py = r'''
from players.player import Player
from MetropolisHastings.shared_weights_DSL import SharedWeightsDSL

class Script_{0}(Player):
\tdef get_action(self, state):
\t\timport numpy as np
\t\tactions = state.available_moves()
\t\tscore_yes_no = 0
\t\tscore_columns = np.zeros(len(actions))
'''
        
        self.program = r'''{0}'''
    def generate_text(self, file_name):
        py = self._py.format(file_name)
        py += self.program.format(self.program_string)
        py += self.program.format('\n')
        py += self.program.format(self.program_column)
        py = codecs.decode(py, 'unicode_escape')
        return py

    def generate_player(self, file_tag):
        class_code = self.generate_text('dynamic')
        # Create file in disk
        file_name = "script_aux_" + str(file_tag) + ".py"
        with open(self.path + '/' + file_name, "w") as file:
            file.write(class_code)
        # Now import it and the class will now be (correctly) seen in __main__
        package = self.path + ".script_aux_" + str(file_tag)
        class_name = "Script_dynamic"
        # This is the programmatically version of 
        # from <package> import <class_name>
        class_name = getattr(__import__(package, fromlist=[class_name]), class_name)
        return class_name()

    def save_file_custom(self, path, file_name):
        py = self.generate_text(file_name)
        with open(path + 'Script_'+ file_name + '.py', 'w') as f:
            f.write(py)
