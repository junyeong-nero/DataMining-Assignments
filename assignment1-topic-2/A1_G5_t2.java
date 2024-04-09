import java.util.*;

public class A1_G5_t2 {
    public static void main(String[] args) {
        
        // default
        String fileDir = "data/groceries.csv";
        float minsup = 0.05f;
        boolean debug = true;

        if (args.length >= 1)
            fileDir = args[0];
        if (args.length >= 2)
            minsup = Float.parseFloat(args[1]);
        if (args.length >= 3)
            debug = Boolean.parseBoolean(args[2]);

        // convert csv file to transaction data        
        Map<Set<String>, Integer> dataset = CSVReader.convert2array(fileDir);
        
        long startTime = System.currentTimeMillis();
        
        // size of transactions
        float T = 0;
        for (int value : dataset.values())
            T += value;

        // generate FP with FP-growth Algorithm
        Map<Set<String>, Integer> freqPatterns = FPGrowth.fpgrowth(dataset, minsup * T);
        
        // print FP
        // for (Set<String> key : freqPatterns.keySet()) {
        //     System.out.println(key + " " + freqPatterns.get(key) / T);
        // }

        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        if (debug)
            System.out.println("Execution time: " + executionTime + " milliseconds");
    }
}


