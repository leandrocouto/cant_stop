#from Script import Script
#from DSL import DSL
#import random
#import sys

#sys.path.insert(0,'..')
import random
import codecs




class DSL:
    """
    Implementation of a Domain Specific Language (DSL) for the Can't Stop
    domain.
    """
    def __init__(self, start):
        self.start = start
        
        self._grammar = {}
        self._grammar[self.start] = [r"for a in actions : \n \t if a in ['y','n'] : \n \t \t forced_condition_if \n \t else : \n \t \t forced_condition_else"]
        self._grammar['forced_condition_if']   = [r"if BOOL_0 : \n \t \t \t return a \n \t \t condition_string"]
        self._grammar['forced_condition_else'] = [r"if BOOL_1 : \n \t \t \t return a \n \t \t condition_numeric"]
        self._grammar['condition_string'] = [r"if BOOL_0 : \n \t \t \t return a \n \t \t condition_string", ""]
        self._grammar['condition_numeric'] = [r"if BOOL_1 : \n \t \t \t return a \n \t \t condition_numeric", ""]
        self._grammar['BOOL_0'] = ["B_0", "B_0 and BOOL_0", "B_0 or BOOL_0"]
        self._grammar['BOOL_1'] = ["B_1", "B_1 and BOOL_1", "B_1 or BOOL_1"]
        self._grammar['B_0'] = [# Strictly "string" actions
                                'DSL.is_stop_action(a)',
                                # All types of actions
                                'DSL.get_player_score(state) > SCORE',
                                'DSL.get_opponent_score(state) > SCORE',
                                'DSL.number_of_neutral_markers_remaining(state) == SMALL_NUM',
                                'DSL.number_cells_advanced_this_round(state) > COLS',
                                'DSL.columns_won_by_opponent(state) == SMALL_NUM',
                                'DSL.columns_won_by_player(state) == SMALL_NUM',
                                'DSL.has_won_column_current_round(state)'
                                ]
        self._grammar['B_1'] = [# Strictly "numeric" actions
                                'DSL.is_doubled_action(a)', 
                                'DSL.is_action_a_column_border(a)', 
                                'DSL.has_won_column_current_round(state)', 
                                'DSL.is_column_in_action(a, COLS )',
                                'DSL.action_wins_at_least_one_column(state,a)',
                                # All types of actions
                                'DSL.number_cells_advanced_this_round_for_col(state, COLS ) > SMALL_NUM',
                                'DSL.number_positions_conquered(state, COLS ) > SMALL_NUM',
                                'DSL.columns_won_by_opponent(state) == SMALL_NUM',
                                'DSL.columns_won_by_player(state) == SMALL_NUM',
                                'DSL.number_of_neutral_markers_remaining(state) == SMALL_NUM',
                                'DSL.number_cells_advanced_this_round(state) > COLS',
                                'DSL.get_player_score(state) > SCORE',
                                'DSL.get_opponent_score(state) > SCORE'
                                ]
        self._grammar['SCORE'] = ['5', '10', '15', '20', '30', '40', '50', '60', '70']
        self._grammar['COLS'] = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        self._grammar['SMALL_NUM'] = ['0', '1', '2', '3']

class Node:
    def __init__(self, node_id, value, is_terminal, parent):
        self.node_id = node_id
        self.value = value
        self.children = []
        self.is_terminal = is_terminal
        self.parent = parent

class ParseTree:
    def __init__(self, dsl, max_nodes):
        self.dsl = dsl
        self.root = Node(node_id = 0, value = self.dsl.start, is_terminal = False, parent = '')
        self.max_nodes = max_nodes
        self.current_id = 0

    def build_tree(self, start_node):
        if self.current_id > self.max_nodes:
            self._finish_tree(self.root)
            return
        else:
            self._expand_children(start_node, self.dsl._grammar[start_node.value])
            for child_node in start_node.children:
                if not child_node.is_terminal:
                    self.build_tree(child_node)

    def _update_nodes_ids(self, node):
        node.node_id = self.current_id
        self.current_id += 1
        for child_node in node.children:
            self._update_nodes_ids(child_node)

    def _expand_children(self, parent_node, dsl_entry):
        dsl_children_chosen = random.choice(dsl_entry)
        children = self._tokenize_dsl_entry(dsl_children_chosen)
        for child in children:
            is_terminal = self._is_terminal(child)
            node_id = self.current_id + 1
            self.current_id += 1
            child_node = Node(node_id = node_id, value = child, is_terminal = is_terminal, parent = parent_node.value)
            parent_node.children.append(child_node)

    def _finish_tree(self, start_node):
        tokens = self._tokenize_dsl_entry(start_node.value)
        is_node_finished = True
        finishable_nodes = [
                            'B_0', 'B_1', 'SCORE', 'COLS', 'SMALL_NUM',
                            'BOOL_0', 'BOOL_1', 'forced_condition_if', 
                            'forced_condition_else'
                            ]
        for token in tokens:
            if token in finishable_nodes and len(start_node.children) == 0:
                is_node_finished = False
                break
        if not is_node_finished:
            self._expand_children(start_node, self.dsl._grammar[start_node.value])
        for child_node in start_node.children:
            self._finish_tree(child_node)

    def _is_terminal(self, dsl_entry):
        tokens = self._tokenize_dsl_entry(dsl_entry)
        for token in tokens:
            if token in self.dsl._grammar:
                return False
        return True

    def generate_program(self):
        list_of_nodes = []
        self._get_traversal_list_of_nodes(self.root, list_of_nodes)
        whole_program = ''
        for node in list_of_nodes:
            whole_program += ' ' + node[0].value
        # Transform '\n' and '\t' into actual new lines and tabs
        whole_program = codecs.decode(whole_program, 'unicode_escape')
        return whole_program

    def _get_traversal_list_of_nodes(self, node, list_of_nodes):
        for child in node.children:
            # Only the values from terminal nodes are relevant for the program
            # synthesis (using a parse tree)
            if child.is_terminal:
                list_of_nodes.append((child, child.parent))
            self._get_traversal_list_of_nodes(child, list_of_nodes)

    def print_tree(self, node, indentation):
        if indentation == '  ':
            print(node.value, ', id = ', node.node_id, ', node parent = ', node.parent)
        else:
            print(indentation, node.value, ', id = ', node.node_id, ', node parent = ', node.parent)
        for child in node.children:
            self.print_tree(child, indentation + '    ')

    def _tokenize_dsl_entry(self, dsl_entry):
        return dsl_entry.split()

    def mutate_tree(self):
        while True:
            index_node = random.randint(1, self.current_id - 1)
            print('index = ', index_node)
            node_to_mutate = self.find_and_mutate(self.root, index_node)
            if node_to_mutate == None:
                self.print_tree(self.root, '  ')
                raise Exception('Node randomly selected does not exist.' + \
                            ' Index sampled = ', index_node + \
                            ' Tree is printed above.'
                            )
            if not node_to_mutate.is_terminal:
                break
        # delete its children and mutate it with new values
        node_to_mutate.children = []
        self.build_tree(node_to_mutate)
        self._finish_tree(self.root)
        self.current_id = 0
        self._update_nodes_ids(self.root)

    def find_and_mutate(self, node, index):
        if node.node_id == index:
            return node
        else:
            for child_node in node.children:
                found_node = self.find_and_mutate(child_node, index)
                if found_node:
                    return found_node
        return None

'''
dsl = DSL('S')
tree = ParseTree(dsl, 100)
tree.build_tree(tree.root)
print()
tree.print_tree(tree.root, '  ')
print()
program = tree.generate_program()
print('program inicial\n')
print(program)
tree.print_tree(tree.root, '  ')
program = tree.generate_program()
print(program)
tree.mutate_tree()
print()
print('arvore dps de mutar')
print()
tree.print_tree(tree.root, '  ')
print()
program = tree.generate_program()
print('program dps de  mutar\n')
print(program)
#program = tree.generate_program()
#print(program)
'''