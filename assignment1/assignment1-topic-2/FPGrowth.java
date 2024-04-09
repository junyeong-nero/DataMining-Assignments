import java.util.*;
import java.util.stream.Collectors;


/**
 * FPNode Class
 */
class FPNode {
    String item;
    int count;
    FPNode parent;
    Map<String, FPNode> children;
    FPNode next;

    public FPNode(String item, int count, FPNode parent) {
        this.item = item;
        this.count = count;
        this.parent = parent;
        this.children = new HashMap<>();
        this.next = null;
    }
}

public class FPGrowth {

    /**
     * FP-growth:
     *    1. contruct FP-tree from transactions
     *    2. pattern mining with constructed FP-tree
     * 
     * @param transactions: converted transactions data from csv file
     * @param minSupport: minimum support
     * @return frequent patterns with their frequency
     */
    public static Map<Set<String>, Integer> fpgrowth(Map<Set<String>, Integer> transactions, float minSupport) {
        Object[] result = constructTree(transactions, minSupport);
        // FPNode root = (FPNode) result[0];
        Map<String, Object[]> headerTable = (Map<String, Object[]>) result[1];

        Map<Set<String>, Integer> frequentItemsets = new HashMap<>();
        mineTree(headerTable, minSupport, new HashSet<>(), frequentItemsets);
        return frequentItemsets;
    }

    /**
     * Construct FP-tree with transactions
     * 
     * @param transactions: transactions
     * @param minSupport: minimum support
     * @return FP-tree, Header-table
     */
    private static Object[] constructTree(Map<Set<String>, Integer> transactions, float minSupport) {
        Map<String, Integer> frequencyList = new HashMap<>();

        // add transaction data to headerTable
        transactions.forEach((transaction, freq) -> transaction.forEach(item -> frequencyList.merge(item, freq, Integer::sum)));

        // pruning items which frequency is less than minimum support
        frequencyList.entrySet().removeIf(entry -> entry.getValue() < minSupport);

        // if headerTable is empty, it is impossible to construct FP-tree. Therefore, return {null, null}
        if (frequencyList.isEmpty()) return new Object[]{null, null};

        // generate newHeaderTable which map item -> [freq, FP-node]s
        Map<String, Object[]> newHeaderTable = new HashMap<>();
        frequencyList.forEach((k, v) -> newHeaderTable.put(k, new Object[]{v, null}));

        // generate a root node of FP-tree
        FPNode root = new FPNode(null, 0, null);

        // iterate all transactions and expand FP-tree with each transaction
        transactions.forEach((transaction, freq) -> {

            // each transactions are filtered and sorted with F-list
            List<String> sortedTransaction = transaction.stream()
                    .filter(frequencyList::containsKey)
                    .sorted(Comparator.comparingInt((String item) -> (int) frequencyList.get(item)).reversed())
                    .collect(Collectors.toList());

            // with ordered frequent items updateTree
            FPNode currentNode = root;
            for (String item : sortedTransaction) {
                currentNode = updateTree(item, currentNode, newHeaderTable, freq);
            }
        });
        return new Object[]{root, newHeaderTable};
    }

    /**
     * Update the child node of `node` with item
     * 
     * @param item
     * @param node: current node
     * @param headerTable
     * @param update
     * @return
     */
    private static FPNode updateTree(String item, FPNode node, Map<String, Object[]> headerTable, int update) {

        // if current node contains child item node
        if (node.children.containsKey(item)) {
            node.children.get(item).count += update;
        } else {
            // else, we need to make a new node and linked to current node
            FPNode newNode = new FPNode(item, update, node);
            node.children.put(item, newNode);

            // In addition, connect new node to headerTable
            if (headerTable.get(item)[1] == null) {
                // if item has no nodes, allocate new node as head node
                headerTable.get(item)[1] = newNode;
            } else {
                // else add to tail
                updateHeader((FPNode) headerTable.get(item)[1], newNode);
            }
        }
        return node.children.get(item);
    }

    /**
     * Link targetNode to next to end node of currentNode
     * 
     * @param nodeToTest
     * @param targetNode
     */
    private static void updateHeader(FPNode curretNode, FPNode targetNode) {
        while (curretNode.next != null) {
            curretNode = curretNode.next;
        }
        curretNode.next = targetNode;
    }

    /**
     * Pattern Mining with constructed FP-tree, HeaderTable
     * 
     * @param headerTable
     * @param minSupport
     * @param prefix
     * @param frequentItemsets
     */
    private static void mineTree(Map<String, Object[]> headerTable, float minSupport, Set<String> prefix, Map<Set<String>, Integer> frequentItemsets) {

        // check whether header table is null or not
        if (headerTable == null) return;

        // iterate for all items in headerTable
        headerTable.keySet().forEach(baseItem -> {

            // generate a new prefix pattern
            Set<String> newFrequentSet = new HashSet<>(prefix);
            newFrequentSet.add(baseItem);
            
            // add generated prefix pattern to result set
            frequentItemsets.put(newFrequentSet, (Integer) headerTable.get(baseItem)[0]);

            // make a conditional pattern base with baseItem
            Map<Set<String>, Integer> conditionalPatternBase = findPatternBase(baseItem, headerTable);

            // construct conditional FP-tree with conditional pattern base
            Object[] result = constructTree(conditionalPatternBase, minSupport);

            // FPNode conditionalRoot = (FPNode) result[0];
            Map<String, Object[]> conditionalHeader = (Map<String, Object[]>) result[1];
            mineTree(conditionalHeader, minSupport, newFrequentSet, frequentItemsets);
        });
    }

    /**
     * Find a pattern base of `baseItem`
     * 
     * @param baseItem
     * @param headerTable
     * @return
     */
    private static Map<Set<String>, Integer> findPatternBase(String baseItem, Map<String, Object[]> headerTable) {
        Map<Set<String>, Integer> transactions = new HashMap<>();

        // node is a head node of baseItem
        FPNode node = (FPNode) headerTable.get(baseItem)[1];

        // traverse all nodes which linked to head node of baseItem
        while (node != null) {

            // find a path from node to root
            List<String> path = ascendTree(node);

            // prefixPath[0] is always node, therefore slice it and add to transactions (conditional pattern base)
            if (path.size() > 1) {
                transactions.put(new HashSet<>(path.subList(1, path.size())), node.count);
            }

            // move to next node (same baseItem)
            node = node.next;
        }
        return transactions;
    }

    /**
     * Add a path from `node` to root to `prefixPath`
     * 
     * @param node
     * @param prefixPath 
     */
    private static List<String> ascendTree(FPNode node) {
        List<String> path = new ArrayList<>();
        while (node.parent != null) {
            path.add(node.item);
            node = node.parent;
        }
        return path;
    }

    public static void main(String[] args) {
        Map<Set<String>, Integer> transactions = new HashMap<>();
        transactions.put(new HashSet<>(Arrays.asList("f", "a", "c", "d", "g", "i", "m", "p")), 1);
        transactions.put(new HashSet<>(Arrays.asList("a", "b", "c", "f", "l", "m", "o")), 1);
        transactions.put(new HashSet<>(Arrays.asList("b", "f", "h", "j", "o")), 1);
        transactions.put(new HashSet<>(Arrays.asList("b", "c", "k", "s", "p")), 1);
        transactions.put(new HashSet<>(Arrays.asList("a", "f", "c", "e", "l", "p", "m", "n")), 1);

        float minSupport = 0.05f;
        Map<Set<String>, Integer> patterns = fpgrowth(transactions, minSupport);

        System.out.println("Frequent Itemsets:");
        patterns.forEach((itemset, support) -> System.out.println(itemset + " " + ((float) support / transactions.size())));
    }
}