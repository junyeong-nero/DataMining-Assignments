
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class CSVReader {


    /**
     * convert csv file to transaction data format to use in FP-growth algorithm
     * 
     * @param fileDir{String}: direction of target csv file
     * @return transactions which convert itemsets{Set<String>} -> frequency{Integer}
     */

    public static Map<Set<String>, Integer> convert2array(String fileDir) {
        String csvFile = fileDir;
        String line = "";
        String cvsSplitBy = ",";

        Map<Set<String>, Integer> transactions = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            while ((line = br.readLine()) != null) {
                String[] data = line.split(cvsSplitBy);
                Set<String> key = new HashSet<>(Arrays.asList(data));
                transactions.put(key, transactions.getOrDefault(key, 0) + 1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return transactions;
    }

    public static void main(String[] args) {
        Map<Set<String>, Integer> map = convert2array(args[0]);
        for (Set<String> key : map.keySet()) {
            System.err.println(key + "[" + map.get(key) + "]");
        }
    }
}
