import sys
import random
import time
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
from IteratedWidth.parse_tree import ParseTree
from IteratedWidth.DSL import DSL
from IteratedWidth.sketch import Sketch

class IteratedWidth:
    def __init__(self, tree, n_expansions, k):
        self.OPEN = []
        self.CLOSED = []
        self.initial_state = tree
        self.n_expansions = n_expansions
        self.k = k

    def run(self):
        len_open = []
        self.OPEN.append(self.initial_state)
        for i in range(self.n_expansions):
            # State to be expanded - First of the list (Breadth-first search)
            novel_state = self.OPEN.pop(0)
            self.expand_children(novel_state)
            # Sort the OPEN list based on the novelty attribute
            #self.OPEN.sort()
            #print(' i = ', i, 'len open: ', len(self.OPEN))
            len_open.append((i, len(self.OPEN),))
            if not self.OPEN:
                print('No more nodes in OPEN')
                break

        return len_open

    def expand_children(self, novel_state):
        """ 
        Expand all node children and add them to OPEN and calculate the 
        state's novelty.
        """
        tree_leaves = novel_state.get_tree_leaves()
        # List for nodes to be added in self.OPEN
        # We separate it for the novelty calculation to not take into
        # consideration just expanded states
        to_be_added = []
        # Stores all k-pairs in both OPEN and CLOSED lists
        # Used to update the novelty of newly expanded states
        all_k_pairs = self.get_k_pairs()
        for leaf in tree_leaves:
            if leaf.is_terminal:
                continue
            dsl_entries = novel_state.dsl._grammar[leaf.value]
            for entry in dsl_entries:
                copied_tree = novel_state.clone_tree()
                # Reset its novelty score
                copied_tree.novelty = 0
                node = copied_tree.find_node(leaf.node_id)
                copied_tree.expand_specific_node(node, entry)
                # Update this state's novelty according to already seen states
                self.update_state_novelty(copied_tree, all_k_pairs)
                to_be_added.append(copied_tree)

        # Since novel_state is already expanded,  add it to the CLOSED list.
        self.CLOSED.append(novel_state)
        # Add expanded children to OPEN
        # Do not add to OPEN (prune) states that do not have a novelty of
        # self.k
        for state in to_be_added:
            if state.novelty == self.k:
                self.OPEN.append(state)
        
    def get_k_pairs(self):
        """ Return all k_pairs in both OPEN and CLOSED lists. """

        open_k_pairs = []
        closed_k_pairs = []
        for open_node in self.OPEN:
            open_k_pairs = open_k_pairs + open_node.k_pairing_values
        for closed_node in self.CLOSED:
            closed_k_pairs = closed_k_pairs + closed_node.k_pairing_values

        return open_k_pairs + closed_k_pairs

    def update_state_novelty(self, state, all_k_pairs):
        """ Update 'state' novelty based on past observed states. """

        for state_k_pair in state.k_pairing_values:
            if state_k_pair not in all_k_pairs:
                if state.novelty < self.k:
                    state.novelty += 1
                else:
                    # No need to continue checking if it already reached
                    # the novelty threshold (self.k)
                    break
        

if __name__ == "__main__":
    tree_max_nodes = 50
    n_expansions = 1000
    k = 2
    all_open = []
    for i in range(10):
        tree = ParseTree(DSL(), tree_max_nodes, k)
        IW = IteratedWidth(tree, n_expansions, k)
        start = time.time()
        len_open = IW.run()
        all_open.append(len_open)
        print(len_open)
        elapsed_time = time.time() - start
        print('Elapsed time = ', elapsed_time)
    '''
    print(all_open)
    for i in range(10):
        plt.plot([all_open[i][j][0] for j in range(len(all_open[i]))], [all_open[i][j][1] for j in range(len(all_open[i]))])
    plt.title("Iterated Width - k = " + str(k))
    plt.xlabel('Number of expansions')
    plt.ylabel('Size of OPEN list')
    plt.grid()
    axes = plt.gca()
    plt.xticks([x * 100 for x in range(0,11,1)])
    axes.set_ylim([-0.05, 1000.05])
    plt.savefig('IW_' + str(k) + '.png')
    plt.close()
    '''