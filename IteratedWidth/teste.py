import sys
import pickle
import time
import os
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from IteratedWidth.sketch import Sketch


yes_no_dsl = DSL()
yes_no_dsl.set_type_action(True)
tree_max_nodes = 150


tree_string = ParseTree(yes_no_dsl, tree_max_nodes)

tree_string.build_tree(tree_string.root)

current_program_string = tree_string.generate_program()
print('current_program_string')
print(current_program_string)
    