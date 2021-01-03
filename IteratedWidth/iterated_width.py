import sys
import random
import time
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from IteratedWidth.toy_DSL import ToyDSL
from IteratedWidth.sketch import Sketch

class IteratedWidth:
    def __init__(self, tree, n_expansions, k):
        self.OPEN = []
        self.CLOSED = []
        self.initial_state = tree
        self.n_expansions = n_expansions
        self.k = k
        self.all_k_pairs = {}
        for i in range(1, self.k+1):
            self.all_k_pairs[i] = []

    def run(self):
        """ Main routine of the IW algorithm. """

        self.OPEN.append(self.initial_state)
        for i in range(self.n_expansions):
            # State to be expanded - First of the list (Breadth-first search)
            novel_state = self.OPEN.pop(0)
            # Since novel_state will be expanded, add it to the CLOSED list.
            self.CLOSED.append(novel_state)
            children = self.expand_children(novel_state)
            # Add expanded children to OPEN
            for state in children:
                # If state has already been seen, ignore it
                if state in self.CLOSED:
                    continue
                if state in self.OPEN:
                    continue
                # Do not add to OPEN (prune) states that do not have a novelty
                # of at least self.k
                if state.novelty != -1:
                    self.OPEN.append(state)
                    # Add all k-pairs from state into all_k_pairs
                    self.update_k_pairs(state)
            print('After expansion - i = ', i, 'len OPEN = ', len(self.OPEN), 'len CLOSED = ', len(self.CLOSED))
            if not self.OPEN:
                print('No more nodes in OPEN')
                return self.OPEN
        print('len closed = ', len(self.CLOSED))
        return self.OPEN

    def update_k_pairs(self, state):
        """ Add 1->k-pairs from state to all_k_pairs avoiding duplicates."""

        for i in range(1, self.k+1):
            for i_pair in state.k_pairing_values[i]:
                if i_pair not in self.all_k_pairs[i]:
                    self.all_k_pairs[i].append(i_pair)

    def expand_children(self, novel_state):
        """ 
        Expand all node children and add them to OPEN and calculate the 
        state's novelty.
        """
        tree_leaves = novel_state.get_tree_leaves()
        # List for nodes to be added in self.OPEN
        children = []
        for leaf in tree_leaves:
            if leaf.is_terminal:
                continue
            dsl_entries = novel_state.dsl._grammar[leaf.value]
            for entry in dsl_entries:
                copied_tree = novel_state.clone()
                # Reset its novelty score
                copied_tree.novelty = 0
                node = copied_tree.find_node(leaf.node_id)
                copied_tree.expand_specific_node(node, entry)
                # Update this state's novelty according to already seen states
                self.update_state_novelty(copied_tree)
                children.append(copied_tree)

        return children

    def update_state_novelty(self, state):
        """ Update 'state' novelty based on past observed states. """

        for i in range(1, self.k+1):
            for i_pair in state.k_pairing_values[i]:
                if i_pair not in self.all_k_pairs[i]:
                    state.novelty = i
                    return
        state.novelty = -1      

if __name__ == "__main__":
    tree_max_nodes = 50
    n_expansions = 100000
    k = 20

    #tree = ParseTree(DSL(), tree_max_nodes, k, True)
    tree = ParseTree(ToyDSL(), tree_max_nodes, k, True)
    IW = IteratedWidth(tree, n_expansions, k)
    start = time.time()
    open_list = IW.run()
    elapsed_time = time.time() - start
    print('Elapsed time = ', elapsed_time)
    print('len = ', len(open_list))
    '''
    for j in range(len(open_list)):
        print('Program - ', j, 'Novelty = ', open_list[j].novelty)
        print(open_list[j].generate_program())
        open_list[j].print_tree()
        print()
    '''
    