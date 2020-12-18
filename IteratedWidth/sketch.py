import random
import codecs
import sys
from IteratedWidth.string_loader import StringLoader, StringFinder

class Sketch:
    def __init__(self, program_string):
        self.program_string = program_string
        self._py = r'''
from players.player import Player

class Script(Player):
\tdef get_action(self, state):
\t\t import numpy as np
\t\t actions = state.available_moves()
'''
        
        self.program = r'''{0}'''
    def generate_text(self):
        py = self._py
        py += self.program.format(self.program_string)
        py = codecs.decode(py, 'unicode_escape')
        print(py)
        return py

    def save_file_custom(self, path, file_name):
        py = self.generate_text(file_name)
        with open(path + 'Script_'+ file_name + '.py', 'w') as f:
            f.write(py)

    def get_object(self):
        modules = {'my_module': self.generate_text()}
        finder = StringFinder(StringLoader(modules))
        sys.meta_path.append(finder)

        from my_module import Script
        my_object = Script()
        return my_object

