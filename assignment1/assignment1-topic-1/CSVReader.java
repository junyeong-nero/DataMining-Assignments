
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class CSVReader {

    public static Map<Set<String>, Integer> convert2array(String fileDir) {
        // CSV 파일 경로
        String csvFile = fileDir;
        String line = "";
        String cvsSplitBy = ",";

        Map<Set<String>, Integer> transactions = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            // CSV 파일을 한 줄씩 읽어들임
            while ((line = br.readLine()) != null) {
                // 쉼표(,)를 기준으로 문자열을 분리
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
