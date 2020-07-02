from Script import Script
from DSL import DSL
import random
import codecs
import re
#import sys

#sys.path.insert(0,'..')

class Node:
    def __init__(self, node_id, value, is_terminal, parent):
        self.node_id = node_id
        self.value = value
        self.children = []
        self.is_terminal = is_terminal
        self.parent = parent

class ParseTree:
    """ Parse Tree implementation given the self.dsl given. """
    
    def __init__(self, dsl, max_nodes):
        """
        - dsl is the domain specific language used to create this parse tree.
        - max_nodes is a relaxed max number of nodes this tree can hold.
        - current_id is an auxiliary field used for id'ing the nodes. 
        """

        self.dsl = dsl
        self.root = Node(
                        node_id = 0, 
                        value = self.dsl.start, 
                        is_terminal = False, 
                        parent = ''
                        )
        self.max_nodes = max_nodes
        self.current_id = 0

    def build_tree(self, start_node):
        """ Build the parse tree according to the self.dsl rules. """

        if self.current_id > self.max_nodes:
            self._finish_tree(self.root)
            return
        else:
            self._expand_children(
                                start_node, 
                                self.dsl._grammar[start_node.value]
                                )
            for child_node in start_node.children:
                if not child_node.is_terminal:
                    self.build_tree(child_node)

    def _expand_children(self, parent_node, dsl_entry):
        """
        Expand the children of 'parent_node'. Since it is a parse tree, in 
        this implementation we choose randomly which child is chosen for a 
        given node.
        """

        dsl_children_chosen = random.choice(dsl_entry)
        children = self._tokenize_dsl_entry(dsl_children_chosen)
        for child in children:
            is_terminal = self._is_terminal(child)
            node_id = self.current_id + 1
            self.current_id += 1
            child_node = Node(
                            node_id = node_id, 
                            value = child, 
                            is_terminal = is_terminal, 
                            parent = parent_node.value
                            )
            parent_node.children.append(child_node)

    def _finish_tree(self, start_node):
        """ 
        Finish expanding nodes that possibly didn't finish fully expanding.
        This can happen if the max number of tree nodes is reached.
        """

        tokens = self._tokenize_dsl_entry(start_node.value)
        is_node_finished = True
        finishable_nodes = self.dsl.finishable_nodes
        for token in tokens:
            if token in finishable_nodes and len(start_node.children) == 0:
                is_node_finished = False
                break
        if not is_node_finished:
            self._expand_children(
                                start_node, 
                                self.dsl._grammar[start_node.value]
                                )
        for child_node in start_node.children:
            self._finish_tree(child_node)

    def _is_terminal(self, dsl_entry):
        """ 
        Check if the DSL entry is a terminal one. That is, check if the entry
        has no way of expanding.
        """

        tokens = self._tokenize_dsl_entry(dsl_entry)
        for token in tokens:
            if token in self.dsl._grammar:
                return False
        return True

    def _tokenize_dsl_entry(self, dsl_entry):
        """ Return the DSL value by spaces to generate the children. """

        return dsl_entry.split()

    def generate_program(self):
        """ Return a program given by the tree in a string format. """

        list_of_nodes = []
        self._get_traversal_list_of_nodes(self.root, list_of_nodes)
        whole_program = ''
        for node in list_of_nodes:
            # Since what is read from the tree is a string and not a character,
            # We have to convert the newline and tab special characters into 
            # strings.
            # This piece of code is needed for indentation purposes.
            newline = '\\' + 'n'
            tab = '\\' + 't'
            if node[0].value in [newline, tab]:
                whole_program += node[0].value
            else:
                whole_program += node[0].value + ' '
                    
        # Transform '\n' and '\t' into actual new lines and tabs
        whole_program = codecs.decode(whole_program, 'unicode_escape')

        return whole_program

    def _get_traversal_list_of_nodes(self, node, list_of_nodes):
        """
        Add to list_of_nodes the program given by the tree when traversing
        the tree in a preorder manner.
        """

        for child in node.children:
            # Only the values from terminal nodes are relevant for the program
            # synthesis (using a parse tree)
            if child.is_terminal:
                list_of_nodes.append((child, child.parent))
            self._get_traversal_list_of_nodes(child, list_of_nodes)

    def mutate_tree(self):
        """ Mutate a single node of the tree. """

        while True:
            index_node = random.randint(1, self.current_id - 1)
            node_to_mutate = self.find_node(self.root, index_node)
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
        # Build the tree again from this node (randomly as if it was creating
        # a whole new tree)
        self.build_tree(node_to_mutate)
        # Finish the tree with possible unfinished nodes (given the max_nodes
        # field)
        self._finish_tree(self.root)
        # Update the nodes ids. This is needed because when mutating a single 
        # node it is possible to create many other nodes (and not just one).
        # So a single id swap is not possible. This also prevents id "holes"
        # for the next mutation, this way every index sampled will be in the
        # tree.
        self.current_id = 0
        self._update_nodes_ids(self.root)

    def find_node(self, node, index):
        """ Return the tree node with the corresponding id. """

        if node.node_id == index:
            return node
        else:
            for child_node in node.children:
                found_node = self.find_node(child_node, index)
                if found_node:
                    return found_node
        return None

    def _update_nodes_ids(self, node):
        """ Update all tree nodes' ids. Used after a tree mutation. """

        node.node_id = self.current_id
        self.current_id += 1
        for child_node in node.children:
            self._update_nodes_ids(child_node)

    def generate_player(self, program, k, n_iterations, tree_max_nodes):
        """ Generate a Player object given the program string. """

        script = Script(program, k, n_iterations, tree_max_nodes)
        return self._string_to_object(script._generateTextScript())

    def _string_to_object(self, str_class, *args, **kwargs):
        """ Transform a program written inside str_class to an object. """
        exec(str_class)
        class_name = re.search("class (.*):", str_class).group(1).partition("(")[0]
        return locals()[class_name](*args, **kwargs)

    def print_tree(self, node, indentation):
        """ Prints the tree in a simplistic manner. Used for debugging. """

        #For root
        if indentation == '  ':
            print(
                node.value, 
                ', id = ', node.node_id, 
                ', node parent = ', node.parent
                )
        else:
            print(
                indentation, 
                node.value, 
                ', id = ', node.node_id, 
                ', node parent = ', node.parent
                )
        for child in node.children:
            self.print_tree(child, indentation + '    ')
'''
dsl = DSL('S')
tree = ParseTree(dsl, 200)
tree.build_tree(tree.root)
tree.print_tree(tree.root, '  ')
print()
program = tree.generate_program() 
print(program)
print()
script = Script(program, 1,2,3)
#print(script)
script.saveFile('')
'''