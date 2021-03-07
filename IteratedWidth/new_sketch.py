import random
import codecs
import sys
from IteratedWidth.string_loader import StringLoader, StringFinder
from IteratedWidth.new_DSL import NewDSL

class NewSketch:
    def __init__(self, program_string):
        self.program_string = program_string
        self._header_py = r'''
from players.player import Player
from IteratedWidth.new_DSL import NewDSL

class Script(Player):
\tdef get_action(self, state):
\t\timport numpy as np
\t\tactions = state.available_moves()
\t\tscores = np.zeros(len(actions))
\t\tif 'y' in actions:
'''
        
        self.program = r'''{0}'''

        self._end_py = r'''
\t\telse:
\t\t\tscores = np.zeros(len(actions))
\t\t\tmove_value = [0, 0, 7, 0, 2, 0, 4, 3, 4, 0, 2, 0, 7]
\t\t\tfor i in range(len(scores)):
\t\t\t\tfor column in actions[i]:
\t\t\t\t\tscores[i] += NewDSL.advance(actions[i]) * move_value[column] - 6 * NewDSL.is_new_neutral(column, state)
\t\t\tchosen_action = actions[np.argmax(scores)]
\t\t\treturn chosen_action
'''
    def generate_text(self):
        py = self._header_py
        py += self.program.format(self.program_string)
        py += self._end_py
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
        #print(str_class)
        exec(str_class)
        class_name = re.search("class (.*):", str_class).group(1).partition("(")[0]
        return locals()[class_name](*args, **kwargs)

    def get_object(self):
        """ Generate an object of the class inside the string self._py """
        return self._string_to_object(self.generate_text())
    

