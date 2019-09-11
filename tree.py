from anytree import NodeMixin, RenderTree, find

class Node(NodeMixin):
    def __init__(self, state, parent=None):
        """- n_visits is the number of visits in this node
        - n_a is a dictionary {key, value} where key is the action taken from this node
        and value is the number of times this action was chosen.
        - q_a is a dictionary {key, value} where key is the action taken from this node
        and value is the mean reward of the simulation that passed through this
        node and used action 'a'.
        - parent is a Node object. 'None' if this node is the root of the tree.
        - children_list is a list of Node.
        """
        super(Node, self).__init__()
        self.state = state
        self.n_visits = 0
        self.n_a = {}
        self.q_a = {}
        self.parent = parent
        self.children_list = []

class Tree:
    def __init__(self, state):
        self.root = Node(state)
    def add(self, node_to_add, parent):
        parent_in_tree = find(self.root, lambda node: node == parent)
        parent_in_tree.children_list.append(node_to_add)
    def get(self, node_to_get):
        #TODO: exception handling
        node_in_tree = find(self.root, lambda node: node == node_to_get)
        return node_in_tree
#my0 = Node('my0')
#my1 = Node('my1', parent=my0)
#my2 = Node('my2', parent=my1)
#my3 = Node('my3', parent=my2)
#my4 = Node('my4', parent=my3)
#my5 = Node('my5', parent=my2)

tree = Tree(0)
tree.add(Node(1, tree.root), tree.root)
tree.add(Node(2, tree.root), tree.root)
tree.add(Node(3, tree.root), tree.root)
tree.add(Node(4, tree.root), tree.root)
tree.add(Node(5, tree.root), tree.root)
for pre, _, node in RenderTree(tree.root):
    treestr = u"%s%s" % (pre, node.state)
    print(treestr.ljust(8))