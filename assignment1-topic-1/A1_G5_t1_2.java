import java.util.*;
import java.util.stream.Collectors;

public class A1_G5_t1_2 {
    public static void main(String[] args) {
        
        String fileDir = "groceries.csv";
        float minsup = 0.05f;
        if (args.length >= 1) {
            fileDir = args[0];
        }
        if (args.length >= 2) {
            minsup = Float.parseFloat(args[1]);
        }

        // Assuming CSVReader.convert2array returns Map<Set<String>, Integer>
        Map<Set<String>, Integer> dataset = CSVReader.convert2array(fileDir);

        // Prepare the transactions for BruteForce
        List<Set<String>> transactions = new ArrayList<>();
        dataset.forEach((itemSet, count) -> {
            for (int i = 0; i < count; i++) {
                transactions.add(new HashSet<>(itemSet));
            }
        });

        // Start timing
        long startTime = System.nanoTime();

        // Instantiate and use BruteForce
        BruteForce bruteForce = new BruteForce(transactions, minsup);
        Map<Set<String>, Double> frequentItemsets = bruteForce.findFrequentItemsets();
        
        // Stop timing
        long endTime = System.nanoTime();
        
        // Calculate runtime in milliseconds
        long duration = (endTime - startTime) / 1_000_000; // Convert from nanoseconds to milliseconds
        
        System.out.println("Found " + frequentItemsets.size() + " frequent itemsets."); // for debug
        System.out.println("Runtime: " + duration + " milliseconds"); // Print the runtime

        // Sort the frequent itemsets by support values
        List<Map.Entry<Set<String>, Double>> sortedItemsets = new ArrayList<>(frequentItemsets.entrySet());
        sortedItemsets.sort(Map.Entry.comparingByValue(Comparator.reverseOrder())); // To sort in descending order of support

        // Print the sorted frequent itemsets in the specified format
        for (Map.Entry<Set<String>, Double> entry : sortedItemsets) {
            // Join the items in the set with a comma
            String itemset = String.join(", ", entry.getKey());
            // Format the support value to 8 decimal places
            String supportValue = String.format("%.8f", entry.getValue());
            System.out.println(itemset + "\t" + supportValue);
        }
    }
}
