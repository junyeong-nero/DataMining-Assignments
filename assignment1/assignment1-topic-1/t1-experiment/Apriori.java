import java.util.*;
import java.util.stream.Collectors;

public class Apriori {
    private List<Set<String>> transactions;
    private double minSupport;
    
    // Constructor to initialize Apriori with transactions and minimum support
    public Apriori(List<Set<String>> transactions, double minSupport) {
        this.transactions = transactions;
        this.minSupport = minSupport;
    }

    // Main method to find all frequent itemsets given the transactions and minSupport
    public Map<Set<String>, Double> findFrequentItemsets() {
        // Result map to store frequent itemsets with their support values
        Map<Set<String>, Double> result = new HashMap<>();
        // Temporary storage to keep count of each itemset
        Map<Set<String>, Integer> itemSetCount = new HashMap<>();
        // Extract all unique items across all transactions
        Set<String> initialSet = new HashSet<>();
        transactions.forEach(initialSet::addAll);
        Set<Set<String>> currentCandidates = initialSet.stream().map(Collections::singleton).collect(Collectors.toSet());

        // Iteratively find frequent itemsets of increasing size
        int k = 1;
        while (!currentCandidates.isEmpty()) {
            Map<Set<String>, Integer> currentCount = countSupport(currentCandidates);

            // Filtering candidates based on minSupport
            currentCandidates = currentCount.entrySet().stream()
                    .filter(entry -> (double) entry.getValue() / transactions.size() >= minSupport)
                    .peek(entry -> result.put(entry.getKey(), (double) entry.getValue() / transactions.size()))
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toSet());

            // Generating next level candidates (k-itemsets)
            currentCandidates = generateNextLevelCandidates(currentCandidates);
            k++;
        }
        return result;
    }

    // Helper method to count the support of each candidate itemset
    private Map<Set<String>, Integer> countSupport(Set<Set<String>> candidates) {
        Map<Set<String>, Integer> count = new HashMap<>();
        for (Set<String> transaction : transactions) {
            for (Set<String> candidate : candidates) {
                if (transaction.containsAll(candidate)) {
                    count.put(candidate, count.getOrDefault(candidate, 0) + 1);
                }
            }
        }
        return count;
    }

    // Helper method to generate next level candidates by combining current candidates
    private Set<Set<String>> generateNextLevelCandidates(Set<Set<String>> currentCandidates) {
        Set<Set<String>> nextLevelCandidates = new HashSet<>();
        List<Set<String>> candidateList = new ArrayList<>(currentCandidates);
        // Combine sets to form new candidates
        for (int i = 0; i < candidateList.size(); i++) {
            for (int j = i + 1; j < candidateList.size(); j++) {
                Set<String> newCandidate = new HashSet<>(candidateList.get(i));
                newCandidate.addAll(candidateList.get(j));
                if (newCandidate.size() == candidateList.get(i).size() + 1) {
                    nextLevelCandidates.add(newCandidate);
                }
            }
        }
        return nextLevelCandidates;
    }

    
    
}

