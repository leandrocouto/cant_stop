import random
import codecs
import sys
from IteratedWidth.string_loader import StringLoader, StringFinder
from IteratedWidth.toy_DSL import ToyDSL

class Sketch:
    def __init__(self, program_string):
        self.program_string = program_string
        self._py = r'''
from players.player import Player
from IteratedWidth.toy_DSL import ToyDSL

class Script(Player):
\tdef get_action(self, state):
\t\t import numpy as np
\t\t actions = state.available_moves()
\t\t scores = np.zeros(len(actions))
\t\t if 'y' in actions:
\t\t\t if ToyDSL.will_player_win_after_n(state):
\t\t\t\t return 'n'
\t\t\t elif ToyDSL.are_there_available_columns_to_play(state):
\t\t\t\t return 'y'
\t\t\t else:
\t\t\t\t score = ToyDSL.calculate_score(state, [7, 7, 3, 2, 2, 1]) + ToyDSL.calculate_difficulty_score(state, 7, 1, 6, 5)
\t\t\t\t if score >= 29:
\t\t\t\t\t return 'n'
\t\t\t\t else:
\t\t\t\t\t return 'y'
\t\t else:
'''
        
        self.program = r'''{0}'''
    def generate_text(self):
        py = self._py
        py += self.program.format(self.program_string)
        py = codecs.decode(py, 'unicode_escape')
        return py

    def save_file_custom(self, path, file_name):
        py = self.generate_text()
        with open(path + 'Script_'+ file_name + '.py', 'w') as f:
            f.write(py)
    '''
    def get_object(self):
        modules = {'my_module': self.generate_text()}
        finder = StringFinder(StringLoader(modules))
        sys.meta_path.append(finder)

        from my_module import Script
        my_object = Script()
        return my_object
    ''' 
    
    
    def _string_to_object(self, str_class, *args, **kwargs):
        """ Transform a program written inside str_class to an object. """
        import re
        exec(str_class)
        class_name = re.search("class (.*):", str_class).group(1).partition("(")[0]
        return locals()[class_name](*args, **kwargs)

    def get_object(self):
        """ Generate an object of the class inside the string self._py """

        return self._string_to_object(self.generate_text())
    

