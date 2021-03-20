import sys
import random
import time
import matplotlib.pyplot as plt
sys.path.insert(0,'..')

class BFS:
    def __init__(self, initial_state, n_states):
        self.OPEN = []
        self.CLOSED = []
        self.initial_state = initial_state
        self.n_states = n_states

    def run(self):
        """ Main routine of the BFS algorithm. """

        self.OPEN.append(self.initial_state)
        self.CLOSED.append(self.initial_state)
        i = 0
        while len(self.CLOSED) < self.n_states:
            print('i = ', i, 'len OPEN = ', len(self.OPEN), 'len CLOSED = ', len(self.CLOSED))
            # State to be expanded - First of the list (Breadth-first search)
            state = self.OPEN.pop(0)
            #children = self.expand_children(state)
            children = self.new_expand_children(state)
            # Add expanded children to OPEN
            for child in children:
                # If state has already been seen, ignore it
                if child in self.CLOSED:
                    continue
                child.depth = state.depth + 1
                self.OPEN.append(child)
                self.CLOSED.append(child)
            i += 1
            if not self.OPEN:
                print('No more nodes in OPEN')
                return self.CLOSED

        return self.CLOSED

    def expand_children(self, state):
        """ Expand all children of state. """

        tree_leaves = state.get_tree_leaves()
        # List for nodes to be added in self.OPEN
        children = []
        for leaf in tree_leaves:
            if leaf.is_terminal:
                continue
            dsl_entries = state.dsl._grammar[leaf.value]
            for entry in dsl_entries:
                copied_tree = state.clone()
                node = copied_tree.find_node(leaf.node_id)
                copied_tree.expand_specific_node(node, entry)
                children.append(copied_tree)

        return children

    def new_expand_children(self, state):
        """ Expand a child of state. """

        tree_leaves = state.get_tree_leaves()
        # List for nodes to be added in self.OPEN
        children = []
        for leaf in tree_leaves:
            if leaf.is_terminal:
                continue
            dsl_entries = state.dsl._grammar[leaf.value]
            for entry in dsl_entries:
                copied_tree = state.clone()
                node = copied_tree.find_node(leaf.node_id)
                copied_tree.expand_specific_node(node, entry)
                children.append(copied_tree)
            break

        return children