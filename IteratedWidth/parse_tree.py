import random
import codecs
import pickle

class Node:
    def __init__(self, node_id, value, is_terminal, parent):
        self.node_id = node_id
        self.value = value
        self.children = []
        self.is_terminal = is_terminal
        self.parent = parent
        self.can_mutate = True

class ParseTree:
    """ Parse Tree implementation given the self.dsl given. """
    
    def __init__(self, dsl, max_nodes, k, is_IW):
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
                        parent = None
                        )
        self.max_nodes = max_nodes
        self.current_id = 0
        self.novelty = 0
        self.k = k
        self.depth = 0
        self.is_IW = is_IW
        self.k_pairing = {}
        self.k_pairing_values = {}
        for i in range(1, self.k+1):
            self.k_pairing[i] = []
            self.k_pairing_values[i] = []
        self.program = 'S'

    '''
    def __eq__(self, other):
        """Overrides the default implementation. """
        
        if isinstance(other, ParseTree):
            return self.program == other.program
        return False
    '''
    
    def set_fixed_nodes(self):
        list_of_nodes = self.get_all_traversal_list_of_nodes()
        for node in list_of_nodes:
            node.can_mutate = False

    def __eq__(self, other):
        """Overrides the default implementation. """
        
        if not isinstance(other, ParseTree):
            return False
        else:
            return self.tree_equality(self.root, other.root)
        return False
    
    def tree_equality(self, node_1, node_2):
        if len(node_1.children) != len(node_2.children):
            return False
        elif node_1.value != node_2.value:
            return False
        else:
            for i in range(len(node_1.children)):
                if not self.tree_equality(node_1.children[i], node_2.children[i]):
                    return False
        return True

    def is_finished(self):
        """ Check if the current tree is complete (no non-terminal nodes). """
        
        for word in self.program.split():
            if word in self.dsl.finishable_nodes:
                return False
        return True
    
    def build_random_tree(self, start_node):
        """ Build the parse tree according to the self.dsl rules. """

        if self.current_id > self.max_nodes:
            self._finish_tree(self.root)
            return
        else:
            self._expand_children(
                                start_node, 
                                self.dsl._grammar[start_node.value]
                                )

            # It does not expand in order, so the tree is able to grow not only
            # in the children's ordering
            index_to_expand = [i for i in range(len(start_node.children))]
            random.shuffle(index_to_expand)

            for i in range(len(index_to_expand)):
                if not start_node.children[i].is_terminal:
                    self.build_random_tree(start_node.children[i])

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
                            parent = parent_node
                            )
            parent_node.children.append(child_node)

    def update_k_pairing(self):
        self.k_pairing = {}
        self.k_pairing_values = {}
        all_paths = self.get_all_paths()

        # Instantiate the lists for the k_pairings
        for i in range(1, self.k+1):
            self.k_pairing[i] = []
            self.k_pairing_values[i] = []

        # Special case - root (S)
        if not self.root.children:
            self.k_pairing[1].append([self.root])
            self.k_pairing_values[1].append(['S'])
            return
        # Special case - 2 nodes (S - statement)
        if len(self.root.children) == 2 and not self.root.children[1].children:
            self.k_pairing[1].append([self.root])
            self.k_pairing[1].append([self.root.children[1]])
            self.k_pairing_values[1].append(['S'])
            self.k_pairing_values[1].append(['statement'])
            return
        # Collect the k-pairs from each path
        for path in all_paths:
            for i in range(1, self.k+1):
                i_pairing = []
                i_pairing_values = []
                for j in range(len(path) - i + 1):
                    i_pairing.append([path[n] for n in range(j, j+i)])
                    i_pairing_values.append([path[n].value for n in range(j, j+i)])
                self.k_pairing[i] += i_pairing
                self.k_pairing_values[i] += i_pairing_values

        for key in self.k_pairing_values:
            no_dupe_pairing = []
            for elem in self.k_pairing_values[key]:
                if elem not in no_dupe_pairing:
                    no_dupe_pairing.append(elem)
            self.k_pairing_values[key] = no_dupe_pairing

    def expand_specific_node(self, node, dsl_entry):
        """
        Expand only one child of 'node' that is not inside the
        already_expanded list.
        """
        children = self._tokenize_dsl_entry(dsl_entry)
        for child in children:
            is_terminal = self._is_terminal(child)
            node_id = self.current_id + 1
            self.current_id += 1
            child_node = Node(
                            node_id = node_id, 
                            value = child, 
                            is_terminal = is_terminal, 
                            parent = node
                            )
            node.children.append(child_node)
        # Update k-pairing only if IW is running this tree
        if self.is_IW:
            # Update the k-pairings
            self.update_k_pairing()
        # Generate the program generated by this tree
        self.program = self.generate_program()

    def _expand_children_finish_tree(self, parent_node):
        """
        "Quickly" expands the children of 'parent_node'. Avoids expanding nodes
        that are recursive. i.e.: A -> A + B 
        """

        #Only "quickly" finish after the max_node threshold has been reached
        if self.current_id < self.max_nodes:
            dsl_children_chosen = random.choice(self.dsl._grammar[parent_node.value])
        else:
            dsl_children_chosen = random.choice(self.dsl.quickly_finish[parent_node.value])
        children = self._tokenize_dsl_entry(dsl_children_chosen)
        for child in children:
            is_terminal = self._is_terminal(child)
            node_id = self.current_id + 1
            self.current_id += 1
            child_node = Node(
                            node_id = node_id, 
                            value = child, 
                            is_terminal = is_terminal, 
                            parent = parent_node
                            )
            parent_node.children.append(child_node)

    def finish_tree_randomly(self):
        """ 
        Finish expanding nodes that possibly didn't finish fully expanding.
        This can happen if the max number of tree nodes is reached.
        """
        
        self._finish_tree(self.root)

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
            self._expand_children_finish_tree(start_node)
        for child_node in start_node.children:
            self._finish_tree(child_node)

    def _is_terminal(self, child):
        """ 
        Check if the DSL entry is a terminal one. That is, check if the entry
        has no way of expanding.
        """
        if child in self.dsl._grammar:
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

        # Manually swap the indentation tokens into actual indentations
        # I(.) maintains the current indentation
        # I(-) decreases by 1 the current indentation
        # I(+) increases by 1 the current indentation
        initial_indent = 3
        indented_program = ''
        for token in whole_program.split():
            if token == 'I(.)':
                indented_program += '\n' + '\t' * initial_indent
            elif token == 'I(+)':
                initial_indent += 1
                indented_program += '\n' + '\t' * initial_indent
            elif token == 'I(-)':
                initial_indent -= 1
                indented_program += '\n' + '\t' * initial_indent
            else:
                indented_program += token + ' '

        indented_program = codecs.decode(indented_program, 'unicode_escape')

        # Removes blank lines
        final_program = ''
        for line in indented_program.splitlines():
            blank_line = True
            for c in line:
                if c.isalnum():
                    blank_line = False
                    break
            if not blank_line:
                final_program += line + '\n'
        return final_program

    def _get_traversal_list_of_nodes(self, node, list_of_nodes):
        """
        Add to list_of_nodes the program given by the tree when traversing
        the tree in a preorder manner.
        """

        for child in node.children:
            if not child.children:
                list_of_nodes.append((child, child.parent))
            self._get_traversal_list_of_nodes(child, list_of_nodes)

    def get_all_traversal_list_of_nodes(self):
        """
        Add to list_of_nodes the program given by the tree when traversing
        the tree in a preorder manner.
        """
        list_of_nodes = []
        self._get_all_traversal_list_of_nodes(self.root, list_of_nodes)
        return list_of_nodes

    def _get_all_traversal_list_of_nodes(self, node, list_of_nodes):
        """ Helper method for self.get_all_traversal_list_of_nodes. """

        list_of_nodes.append(node)
        for child in node.children:
            # Only the values from terminal nodes are relevant for the program
            # synthesis (using a parse tree)
            self._get_all_traversal_list_of_nodes(child, list_of_nodes)

    def get_tree_leaves(self):
        """ Return a list of leaves (Node objects) of self. """

        leaves = []
        self._get_tree_leaves(self.root, leaves)
        return leaves

    def _get_tree_leaves(self, node, leaves):
        """ Helper method for self.get_tree_leaves. """

        if not node.children:
            if node.value not in ['I(.)', 'I(+)', 'I(-)']:
                leaves.append(node)
        for child in node.children:
            self._get_tree_leaves(child, leaves)

    

    def find_node(self, index):
        """ Return the tree node with the corresponding id. """

        return self._find_node(self.root, index)

    def _find_node(self, node, index):
        """ Helper method for self.find_node. """

        if node.node_id == index:
            return node
        else:
            for child_node in node.children:
                found_node = self._find_node(child_node, index)
                if found_node:
                    return found_node
        return None

    def print_tree(self):
        """ Prints the tree in a simplistic manner. Used for debugging. """
        self._print_tree(self.root, '  ')

    def _print_tree(self, node, indentation):
        """ Helper method for self.print_tree. """

        #For root
        if indentation == '  ':
            print(
                node.value, 
                ', id = ', node.node_id, 
                ', node parent = None'
                )
        else:
            if node.value not in ['I(.)', 'I(+)', 'I(-)']:
                print(
                    indentation, 
                    node.value, 
                    ', id = ', node.node_id, 
                    ', node parent = ', node.parent.value
                    )
        for child in node.children:
            self._print_tree(child, indentation + '    ')

    def print_tree_to_file(self, file, to_print):
        """ Prints the tree in a simplistic manner. Used for debugging. """
        self._print_tree_to_file(self.root, '  ', file, to_print)
        with open(file, 'a') as f:
            print(file=f)

    def _print_tree_to_file(self, node, indentation, file, to_print):
        """ Helper method for self.print_tree. """

        #For root
        if indentation == '  ':
            with open(file, 'a') as f:
                print(to_print, file=f)
                print(
                    node.value, 
                    ', id = ', node.node_id, 
                    ', node parent = None',
                    file=f
                    )
        else:
            with open(file, 'a') as f:
                if node.value not in ['I(.)', 'I(+)', 'I(-)']:
                    print(
                        indentation, 
                        node.value, 
                        ', id = ', node.node_id, 
                        ', node parent = ', node.parent.value,
                        file=f
                        )
        for child in node.children:
            self._print_tree_to_file(child, indentation + '    ', file, to_print)

    def _update_nodes_ids(self, node):
        """ Update all tree nodes' ids. Used after a tree mutation. """

        node.node_id = self.current_id
        self.current_id += 1
        for child_node in node.children:
            self._update_nodes_ids(child_node)

    '''
    def _can_mutate(self, node):
        if node.value in self.dsl.terminal_nodes:
            return True
        else:
            return False
    '''
    
    def _can_mutate(self, node):
        if not node.can_mutate:
            return False
        else:
            if node.value not in ['I(.)', 'I(+)', 'I(-)'] and not node.is_terminal:
                return True
            else:
                return False
        '''
        if node.can_mutate and node.value not in ['I(.)', 'I(+)', 'I(-)'] and not node.is_terminal:
            return True
        else:
            return False
        '''
    
    def mutate_tree(self):
        """ Mutate a single node of the tree. """

        # It can happen that a state cannot be mutated (because of the local
        # search restrictions to not mutate already predefined nodes)
        max_tries = 1000
        while True:
            max_tries -= 1
            index_node = random.randint(0, self.current_id-1)
            node_to_mutate = self.find_node(index_node)
            if node_to_mutate == None:
                self.print_tree()
                raise Exception('Node randomly selected does not exist.' + \
                    ' Index sampled = ' + str(index_node) + \
                    ' Tree is printed above.')
            if self._can_mutate(node_to_mutate):
                break

            if max_tries == 0:
                return

        # delete its children and mutate it with new values
        node_to_mutate.children = []

        # Update the nodes ids. This is needed because when mutating a single 
        # node it is possible to create many other nodes (and not just one).
        # So a single id swap is not possible. This also prevents id "holes"
        # for the next mutation, this way every index sampled will be in the
        # tree.
        self.current_id = 0
        self._update_nodes_ids(self.root)
        # Build the tree again from this node (randomly as if it was creating
        # a whole new tree)
        self.build_random_tree(node_to_mutate)
        # Finish the tree with possible unfinished nodes (given the max_nodes
        # field)
        self.finish_tree_randomly()
        # Updating again after (possibly) finishing expanding possibly not
        # expanded nodes.
        self.current_id = 0
        self._update_nodes_ids(self.root)

    def clone(self):
        """ Return a 'deepcopy' copy of self. """

        return pickle.loads(pickle.dumps(self, -1))

    def get_all_paths(self):
        """
        Calculate all possible paths from each leaf to all other leaves,
        traversing it from left to right. The number of paths is squared to the
        number of leaves in the tree ( (n * (n-1)) / 2 , n being the number of
        leaves).
        Return all paths (a list of Node objects) given the parse tree.
        """

        all_leaves = self.get_tree_leaves()
        #all_leaves = [leaf for leaf in all_leaves if leaf.value not in ['I(.)', 'I(+)', 'I(-)']]
        all_paths = []
        # Iterate through each leaf of the tree to get its paths to other leaves
        for leaf in all_leaves:
            initial_list = [leaf]
            # While it can still go up in the tree to get all upward paths
            while initial_list[-1].parent != None:
                all_downward_paths = []
                # Get all downward paths from the node at the end of the list
                self._get_downward_paths(all_downward_paths, initial_list[-1].parent, [], initial_list[-1])
                # Downward path can output many paths (if it is leveled)
                for i in range(len(all_downward_paths)):
                    all_downward_paths[i] = initial_list + all_downward_paths[i]
                # Append to the 'global' all_paths list
                for path in all_downward_paths:
                    all_paths.append(path)
                # Needed if there are upward nodes (next while iteration)
                initial_list.append(initial_list[-1].parent)

        # Remove  paths containing indentation
        new_all_paths = []
        for path in all_paths:
            invalid = False
            for node in path:
                if node.value in ['I(.)', 'I(+)', 'I(-)']:
                    invalid = True
                    break
            if not invalid:
                new_all_paths.append(path)

        return new_all_paths

    def _get_downward_paths(self, all_downward_paths, node, path, node_called):
        """ 
        Get all paths from node passing through all its children ramifications.
        If a path reaches a leaf, add path to all_downward_paths.
        Otherwise, append the current node to path.
        """

        path.append(node)
        # If it's a leaf, end current path by adding it to all_downward_paths
        if not node.children:
            all_downward_paths.append(path)
        else:
            # Get the index of node_called to avoid adding it again to path.
            # try-catch needed because of recursive calls, node_called might not
            # be in node.children anymore.
            try:
                index = node.children.index(node_called)
            except:
                index = -1
            if index == len(node.children):
                return
            # Start from that position onwards
            for i in range(index+1, len(node.children)):
                if node.children[i] == node_called:
                    continue
                copy_path = pickle.loads(pickle.dumps(path, -1))
                # Get subsequent paths from node's children
                self._get_downward_paths(
                                            all_downward_paths, 
                                            node.children[i], 
                                            copy_path, 
                                            node_called
                                        )

