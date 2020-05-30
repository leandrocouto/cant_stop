from Script import Script
from DSL import DSL
import random
import sys

sys.path.insert(0,'..')

class Node:
    """ Node representation for the DSLTree implementation. """
    def __init__(self, state, parent, is_terminal = False, type_terminal = None):
        """
        - state is a program induced from a DSL.
        - children is a list storing all subsequent states after applying
          a single grammatic transformation on self.state.
        """
        self.state = state
        self.children = []
        self.possible_values = []
        self.is_terminal = is_terminal
        self.type_terminal = type_terminal
        self.parent = parent
        self.node_id = 0
        self.is_children_expanded = False

class DSLTree:
    """ Tree representing all transitions made based on the DSL provided. """
    def __init__(self, node, dsl):
        """
        - root is the starting symbol of the tree.
        - dsl is an instance of the DSL class.
        """
        self.root = node
        self.dsl = dsl
        self.node_id = 0
        self.max_nodes = 30

    def build_tree(self):
        while self.node_id <= self.max_nodes:
            self._build_tree(self.root)
        self.print_tree(self.root, '  ')
        #exit()
        print()
        self._assign_nodes(self.root)
        self.print_tree(self.root, '  ')
        #exit()
        

    def _build_tree(self, node):
        print('node escolhido = ', node.state)
        if node.is_children_expanded:
            if len(node.children) != 0:
                child_to_expand = random.choice(node.children)
                self._build_tree(child_to_expand)
        else:
            self._expand_children(node)
            node.is_children_expanded = True
            if len(node.children) != 0:
    	        child_to_expand = random.choice(node.children)
    	        self._build_tree(child_to_expand)

    def _expand_children(self, node):
        if node.state == 'ROOT':
            node.possible_values.append('if BOOL OP ROOT')
            node.possible_values.append('')

            node_bool = Node('BOOL', 'BOOL')
            self.node_id += 1
            node_bool.node_id = self.node_id
            node.children.append(node_bool)

            node_op = Node('OP', 'OP')
            self.node_id += 1
            node_op.node_id = self.node_id
            node.children.append(node_op)

            node_newrule = Node('ROOT', 'ROOT')
            self.node_id += 1
            node_newrule.node_id = self.node_id
            node.children.append(node_newrule)

        if node.state == 'BOOL':
            node.possible_values.append('B_0')

            node_terminalnum = Node('terminal_num', '',  True, 'SMALL')
            self.node_id += 1
            node_terminalnum.node_id = self.node_id
            node.children.append(node_terminalnum)

            node_terminalsmall = Node('terminal_small', '', True, 'SMALL_NUM')
            self.node_id += 1
            node_terminalsmall.node_id = self.node_id
            node.children.append(node_terminalsmall)

        if node.state == 'OP':
            node.possible_values.append('and BOOL OP')
            node.possible_values.append('or BOOL OP')
            node.possible_values.append('')

            node_bool = Node('BOOL', 'BOOL')
            self.node_id += 1
            node_bool.node_id = self.node_id
            node.children.append(node_bool)

            node_op = Node('OP', 'OP')
            self.node_id += 1
            node_op.node_id = self.node_id
            node.children.append(node_op)

    def _assign_nodes(self, node):
        """Assign to 'node' a value related to its type. """

        if node.is_terminal:
            if node.state == 'terminal_num':
                node.state = random.choice(self.dsl._grammar['COLS'])
                node.parent = 'COLS'
            elif node.state == 'terminal_small':
                node.state = random.choice(self.dsl._grammar['SMALL_NUM'])
                node.parent = 'SMALL_NUM'
        else:
            # Force the program to have at least one rule
            if node.node_id == 0:
                node.state = 'if BOOL OP ROOT'
            elif len(node.possible_values) != 0:
                node.state = random.choice(node.possible_values)
                if node.state in self.dsl._grammar:
                    node.state = random.choice(self.dsl._grammar[node.state])
            # At this point the generated tree might not be finished (due to
            # the number of max nodes provided)
            
            elif len(node.possible_values) == 0:
                # Force to finish completing the node values. Otherwise, we
                # could generate programs like:
                # if DSL.isStopAction(a) or DSL.containsNumber(a, 4) and BOOL OP
                
                # Force to finish 'OP'
                if node.state == 'OP':
                    node.state = ''
                elif node.state == 'BOOL':
                    # Force to finish BOOL
                    node.state = random.choice(self.dsl._grammar[node.state])
                    # Force to finish ['B_0']
                    node.state = random.choice(self.dsl._grammar[node.state])
                    # Force to finish possibles ['NUMBER', 'SMALL_NUM']
                    symbols = node.state.split()
                    for symbol in symbols:
                        if symbol in self.dsl._grammar:
                            node.state = node.state.replace(
                                    symbol, 
                                    random.choice(self.dsl._grammar[symbol]), 
                                    1
                                    )
                elif node.state == 'ROOT':
                    node.state = ''
            for child in node.children:
                self._assign_nodes(child)

    def generate_random_program(self):
        list_of_nodes = []
        self._get_traversal_list_of_nodes(self.root, list_of_nodes)
        program = self.root.state
        for i in range(len(list_of_nodes) -1):
            program = program.replace(
                list_of_nodes[i+1][0].parent, list_of_nodes[i+1][0].state, 1
                                    )
        rules = program.split('if')
        if rules[0] == '':
            del rules[0]
        rules = ['if'+ rule for rule in rules]
        return rules

    def _get_traversal_list_of_nodes(self, node, list_of_nodes):
        # In case node should have children nodes but does not.
        if len(node.children) == 0 and not node.is_terminal:
            symbols = node.state.split()
            for symbol in symbols:
                if symbol in self.dsl._grammar:
                    node.state = node.state.replace(
                                    symbol, 
                                    random.choice(self.dsl._grammar[symbol]), 
                                    1
                                    )
        for child in node.children:
            list_of_nodes.append((child, child.parent))
            self._get_traversal_list_of_nodes(child, list_of_nodes)

    def generate_mutated_program(self, program):
        tree_size = self.get_tree_size(self.root)
        valid_mutation = False
        while not valid_mutation:
            index_node = random.randint(2, tree_size - 2)
            self._mutate_node(self.root, index_node)
            mutated_program = self.generate_random_program()
            if mutated_program != program:
                valid_mutation = True
        return mutated_program

    def _mutate_node(self, node, index):
        if node.node_id == index:
            new_value = random.choice(self.dsl._grammar[node.parent])
            

            if new_value in self.dsl._grammar:
                new_value = random.choice(self.dsl._grammar[new_value])

            node.state = new_value
            return
        for child in node.children:
            self._mutate_node(child, index)
 
    def string_to_object(self, str_class, *args, **kwargs):
        import re
        exec(str_class)
        class_name = re.search("class (.*):", str_class).group(1).partition("(")[0]
        return locals()[class_name](*args, **kwargs)

    def generate_player(self, program):
        script = Script(program, 0)
        script_generated = self.string_to_object(script._generateTextScript())
        return script_generated

    def get_tree_size(self, node):
        n_nodes = 1
        for child in node.children:
            n_nodes += self.get_tree_size(child)
        return n_nodes

    def print_tree(self, node, indentation):
        if indentation == '  ':
            print(node.state, ', parent = ', node.parent, ', id = ', node.node_id)
        else:
            print(indentation, node.state, ', parent = ', node.parent, ', id = ', node.node_id)
        for child in node.children:
            self.print_tree(child, indentation + '    ')

tree = DSLTree(Node('ROOT', ''), DSL())
tree.build_tree()
print(tree.generate_random_program())