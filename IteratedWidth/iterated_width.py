import sys
import random
import time
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree

class IteratedWidth:
    def __init__(self, tree, n_states, k):
        """
          - OPEN is a list of states discovered but not yet expanded.
          - CLOSED is a list that includes all states in OPEN and states that
            have already been expanded.
        """
        self.OPEN = []
        self.CLOSED = []
        self.initial_state = tree
        self.n_states = n_states
        self.k = k
        self.all_k_pairs = {}
        for i in range(1, self.k+1):
            self.all_k_pairs[i] = []

    def run(self):
        """ Main routine of the IW algorithm. """
        open_history = []
        self.OPEN.append(self.initial_state)
        self.CLOSED.append(self.initial_state)
        # Update initial_state k-pairs
        self.initial_state.update_k_pairing()
        self.update_state_novelty(self.initial_state)
        self.update_all_k_pairs(self.initial_state)
        i = 0
        while len(self.CLOSED) < self.n_states:
            # State to be expanded - First of the list (Breadth-first search)
            state = self.OPEN.pop(0)
            children = self.expand_children(state)
            # Add expanded children to OPEN/CLOSED
            for child in children:
                # Prune states that have already been seen
                if child in self.CLOSED:
                    continue
                # Prune states that do not have a novelty of at least self.k
                if child.novelty != -1:
                    child.depth = state.depth + 1
                    self.OPEN.append(child)
                    self.CLOSED.append(child)
                    # Add all k-pairs from state into all_k_pairs
                    self.update_all_k_pairs(child)
            open_history.append((i, len(self.OPEN)))
            print('After expansion - i = ', i, 'len OPEN = ', len(self.OPEN), 'len CLOSED = ', len(self.CLOSED))
            i += 1
            if not self.OPEN:
                print('No more nodes in OPEN')
                return self.CLOSED
        return self.CLOSED

    def update_all_k_pairs(self, state):
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
            break
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
    n_states = 1000
    k = 15
    from IteratedWidth.toy_DSL import ToyDSL
    #tree = ParseTree(DSL(), tree_max_nodes, k, True)
    tree = ParseTree(ToyDSL(), tree_max_nodes, k, True)
    IW = IteratedWidth(tree, n_states, k)
    start = time.time()
    closed_list, _ = IW.run()
    elapsed_time = time.time() - start
    print('Elapsed time = ', elapsed_time)
    print('len = ', len(closed_list))
    print('closed')
    for state in closed_list:
        if state.is_finished():
            print(state.generate_program())
    