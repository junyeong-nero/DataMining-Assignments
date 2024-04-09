import java.util.*;
import java.util.stream.Collectors;

public class BruteForce {
    private List<Set<String>> transactions;
    private double minSupport;

    public BruteForce(List<Set<String>> transactions, double minSupport) {
        this.transactions = transactions;
        this.minSupport = minSupport;
    }

    public Map<Set<String>, Double> findFrequentItemsets() {
        Map<Set<String>, Double> frequentItemsets = new HashMap<>();
        Set<String> allItems = new HashSet<>();
        transactions.forEach(allItems::addAll);
        List<String> itemList = new ArrayList<>(allItems);

        // Generate all possible itemsets
        List<Set<String>> allPossibleItemsets = new ArrayList<>();
        int n = itemList.size();

        System.out.println("Total transactions: " + transactions.size());
        System.out.println("Unique items: " + n);
        System.out.println("Generating " + ((1 << n) - 1) + " possible itemsets");
        
        // Using bit manipulation to generate all subsets
        for (long i = 1; i < (1 << n); i++) {
            Set<String> itemset = new HashSet<>();
            for (int j = 0; j < n; j++) {
                if ((i & (1 << j)) > 0) {
                    itemset.add(itemList.get(j));
                }
            }
            allPossibleItemsets.add(itemset);
        }

        // Count the support for each itemset
        for (Set<String> itemset : allPossibleItemsets) {
            int supportCount = 0;
            for (Set<String> transaction : transactions) {
                if (transaction.containsAll(itemset)) {
                    supportCount++;
                }
            }
            double support = (double) supportCount / transactions.size();
            if (support >= minSupport) {
                frequentItemsets.put(itemset, support);
            }
        }

        return frequentItemsets;
    }
}