from collections import defaultdict

# Data Structure for FP-Tree
class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None
        
### FP-growth, it has two steps
#a
# 1. Generate FP-tree
# 2. Mining frequent patterns with FP-tree
        
def fpgrowth(transactions, min_support):
    # 1. Generate FP-tree
    root, header_table = construct_tree(transactions, min_support)
    frequent_itemsets = {}
    
    # 2. Mining frequent patterns with FP-tree
    mine_tree(header_table, min_support, set(), frequent_itemsets)
    return frequent_itemsets
        
        
### GENERATE FP-Tree

# arugments:
#   transactions: List[List[items]]
#   min_support: Int
#
# returns:
#   FP-tree: FPNode
#   header_table: Dict[items]

def construct_tree(transactions, min_support):
    header_table = defaultdict(int)
    
    # get frequency of items
    size = 0
    for transaction, freq in transactions.items():
        for item in transaction:
            header_table[item] += freq
            size += freq
    
    # Remove items below min support
    header_table = {k: v for k, v in header_table.items() if v / size >= min_support}
    frequent_items = set(header_table.keys())
    
    # if no table entries
    if len(frequent_items) == 0:
        return None, None
    
    # preparing for adding FP-nodes to each header
    # header[item][0] = frequency of item
    # header[item][1] = FP-nodes of item
    for k in header_table:
        header_table[k] = [header_table[k], None]
    
    root = FPNode(None, None, None)
    
    # generate FP-tree from transactions
    for transaction, freq in transactions.items():
        
        # get items contained in frequenct_items
        transaction = [item for item in transaction if item in frequent_items]
        
        # sort items with frequency
        transaction.sort(key=lambda item: header_table[item][0], reverse=True)
        
        # following sorted transactions, update tree
        current_node = root
        for item in transaction:
            current_node = update_tree(item, current_node, header_table, update=freq)
    
    return root, header_table

def update_tree(item, node, header_table, update=1):
    if item in node.children:
        # if already node has children
        # just add 1
        node.children[item].count += update
    else:
        # if not
        #   1. make a new FP-node
        #   2. if the node is a first node of item
        #       just connect the node to header[item]
        #   3. if not,
        #       connect the node to tail of header[item][1] -> implemented with `update_header`
        
        # 1.
        node.children[item] = FPNode(item, update, node)
        if header_table[item][1] is None:
            # 2.
            header_table[item][1] = node.children[item]
        else:
            # 3.
            update_header(header_table[item][1], node.children[item])
    return node.children[item]


def update_header(node_to_test, target_node):
    # find tail node
    while node_to_test.next is not None:
        node_to_test = node_to_test.next
        
    # add target node to tail node
    node_to_test.next = target_node



### 2. Mining frequent patterns with FP-tree

def mine_tree(header_table, min_support, prefix, frequent_itemsets):
    for base_item in header_table.keys():
        
        # make a new patterns with prefix + base_item
        new_frequent_set = prefix.copy()
        new_frequent_set.add(base_item)
        frequent_itemsets[frozenset(new_frequent_set)] = header_table[base_item][0]

        # find prefix path with base_item
        conditional_tree = find_prefix_path(base_item, header_table)
        
        # construct conditional FP-tree and FP-growth~
        conditional_root, conditional_header = construct_tree(conditional_tree, min_support)
        if conditional_header is not None:
            mine_tree(conditional_header, min_support, new_frequent_set, frequent_itemsets)
        
            
# calculate conditional FP-graph
def find_prefix_path(base_item, header_table):
    paths = {}
    
    # FP-nodes from base_item
    node = header_table[base_item][1]
    
    while node is not None:
        prefix_path = []
        ascend_tree(node, prefix_path)
        
        # prefix_path[0] == node.item, always
        if len(prefix_path) > 1:
            paths[frozenset(prefix_path[1:])] = node.count
            
        # iterate with next node that has same base_time
        node = node.next
        
    return paths

# generate ascend tree from node to root.
def ascend_tree(node, prefix_path):
    if node.parent is not None:
        prefix_path.append(node.item)
        ascend_tree(node.parent, prefix_path)




# Sample transactions data
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'eggs'],
    ['bread', 'eggs']
]

# Sample transcations data in paper
transactions = [
    'facdgimp',
    'abcflmo',
    'bfhjo',
    'bcksp',
    'afcelpmn',
]

# transactions = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#            ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#            ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
#            ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
#            ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

transactions_convert = {}
for transaction in transactions:
    transactions_convert[frozenset(transaction)] = transactions_convert.get(frozenset(transaction), 0) + 1


# print(transactions_convert)

# Minimum support threshold
min_support = 0.05

# Perform FP-Growth
patterns = fpgrowth(transactions_convert, min_support)

# Print frequent itemsets
print("Frequent Itemsets:")
for itemset, support in patterns.items():
    print(sorted(itemset), support / len(transactions))
