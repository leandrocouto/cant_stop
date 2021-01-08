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
        """
          - OPEN is a list of states discovered but not yet expanded.
          - CLOSED is a list that includes all states in OPEN and states that
            have already been expanded.
        """
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
        open_history = []
        self.OPEN.append(self.initial_state)
        self.CLOSED.append(self.initial_state)
        for i in range(self.n_expansions):
            # State to be expanded - First of the list (Breadth-first search)
            novel_state = self.OPEN.pop(0)
            children = self.expand_children(novel_state)
            # Add expanded children to OPEN/CLOSED
            for state in children:
                # Prune states that have already been seen
                if state in self.CLOSED:
                    continue
                # Prune states that do not have a novelty of at least self.k
                if state.novelty != -1:
                    self.OPEN.append(state)
                    self.CLOSED.append(state)
                    # Add all k-pairs from state into all_k_pairs
                    self.update_k_pairs(state)
            open_history.append((i, len(self.OPEN)))
            print('After expansion - i = ', i, 'len OPEN = ', len(self.OPEN), 'len CLOSED = ', len(self.CLOSED))
            if not self.OPEN:
                print('No more nodes in OPEN')
                return self.CLOSED, open_history
        return self.CLOSED, open_history

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
    n_expansions = 1000
    k = 15
    
    #tree = ParseTree(DSL(), tree_max_nodes, k, True)
    tree = ParseTree(ToyDSL(), tree_max_nodes, k, True)
    IW = IteratedWidth(tree, n_expansions, k)
    start = time.time()
    closed_list, _ = IW.run()
    elapsed_time = time.time() - start
    print('Elapsed time = ', elapsed_time)
    print('len = ', len(closed_list))
    print('closed')
    for state in closed_list:
        if state.is_finished():
            print(state.generate_program())
    
    '''
    all_open_history = []
    for i in range(1, k):
        #tree = ParseTree(DSL(), tree_max_nodes, k, True)
        tree = ParseTree(ToyDSL(), tree_max_nodes, i, True)
        IW = IteratedWidth(tree, n_expansions, i)
        start = time.time()
        closed_list, open_history = IW.run()
        all_open_history.append(open_history)
        elapsed_time = time.time() - start
        print('Elapsed time = ', elapsed_time)
        print('len = ', len(closed_list))

    for i in range(len(all_open_history)):
        curr_open_history = all_open_history[i]
        x = [data[0] for data in curr_open_history]
        y = [data[1] for data in curr_open_history]
        print('y = ', y)
        plt.plot(x, y, label = "k = " + str(i+1))
    plt.title("OPEN list size per expansion")
    plt.xlabel('Number of expansions')
    plt.ylabel('OPEN list size')
    plt.grid()
    plt.legend()
    plt.savefig('IW_over_k.png')
    plt.close()
    '''
    