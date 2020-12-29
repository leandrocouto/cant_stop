import sys
import random
import time
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from IteratedWidth.sketch import Sketch

class BFS:
    def __init__(self, initial_state, n_expansions):
        self.OPEN = []
        self.CLOSED = []
        self.initial_state = initial_state
        self.n_expansions = n_expansions

    def run(self):
        """ Main routine of the BFS algorithm. """

        self.OPEN.append(self.initial_state)
        for i in range(self.n_expansions):
            print('i = ', i, 'len OPEN = ', len(self.OPEN), 'len CLOSED = ', len(self.CLOSED))
            # State to be expanded - First of the list (Breadth-first search)
            state = self.OPEN.pop(0)
            # Since state will be expanded, add it to the CLOSED list.
            self.CLOSED.append(state)
            children = self.expand_children(state)
            # Add expanded children to OPEN
            for state in children:
                # If state has already been seen, ignore it
                if state in self.CLOSED:
                    continue
                if state in self.OPEN:
                    continue
                self.OPEN.append(state)
            if not self.OPEN:
                print('No more nodes in OPEN')
                return self.OPEN
        return self.OPEN

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
                # Reset its novelty score
                copied_tree.novelty = 0
                node = copied_tree.find_node(leaf.node_id)
                copied_tree.expand_specific_node(node, entry)
                children.append(copied_tree)

        return children

if __name__ == "__main__":
    tree_max_nodes = 50
    n_expansions = 10000
    k = 10
    tree = ParseTree(DSL(), tree_max_nodes, k, False)
    BFS = BFS(tree, n_expansions)
    start = time.time()
    open_list = BFS.run()
    elapsed_time = time.time() - start
    print('Elapsed time = ', elapsed_time)
    print('len = ', len(open_list))