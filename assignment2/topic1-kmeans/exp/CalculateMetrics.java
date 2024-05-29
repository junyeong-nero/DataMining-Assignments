import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CalculateMetrics {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java CalculateMetrics <input_csv_file> <input_cluster_file>");
            return;
        }

        String csvFile = args[0];
        String clusterFile = args[1];

        double[][] points = readCSV(csvFile);
        int[] trueLabels = readTrueLabels(csvFile);
        int[] predictedLabels = readClusters(clusterFile);

        double silhouetteScore = calculateSilhouetteScore(points, predictedLabels);
        double nmi = calculateNMI(trueLabels, predictedLabels);

        System.out.println("Silhouette Score: " + silhouetteScore);
        System.out.println("Normalized Mutual Information (NMI): " + nmi);
    }

    public static double[][] readCSV(String file) {
        List<double[]> pointsList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double x = Double.parseDouble(values[1]);
                double y = Double.parseDouble(values[2]);
                pointsList.add(new double[]{x, y});
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return pointsList.toArray(new double[0][0]);
    }

    public static int[] readTrueLabels(String file) {
        List<Integer> labelsList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                int label = Integer.parseInt(values[3]);
                labelsList.add(label);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labelsList.stream().mapToInt(i -> i).toArray();
    }

    public static int[] readClusters(String file) {
        List<Integer> labelsList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            Map<String, Integer> clusterMap = new HashMap<>();
            int clusterId = 0;

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;  // 빈 줄 건너뛰기
                }
                try {
                    String[] parts = line.split("=>");
                    if (parts.length < 2) {
                        throw new IllegalArgumentException("Invalid line format: " + line);
                    }   
                
                    String[] elements = parts[1].trim().split("\\s+");
                    for (String element : elements) {
                        clusterMap.put(element.trim(), clusterId);
                    }
                    clusterId++;
                } catch (Exception e) {
                    System.err.println("Error processing line: " + line);
                    e.printStackTrace();
                }
            }

            for (int i = 0; i < clusterMap.size(); i++) {
                if (clusterMap.containsKey("p" + (i + 1))) {
                    labelsList.add(clusterMap.get("p" + (i + 1)));
                } else {
                    labelsList.add(-1);  // Missing points are given a default value
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labelsList.stream().mapToInt(i -> i).toArray();
    }



    public static double calculateSilhouetteScore(double[][] points, int[] labels) {
        int n = points.length;
        double[] a = new double[n];
        double[] b = new double[n];

        for (int i = 0; i < n; i++) {
            int cluster = labels[i];
            double[] point = points[i];
            double intraClusterDist = 0.0;
            double nearestClusterDist = Double.MAX_VALUE;
            int intraClusterCount = 0;
            int[] clusterCounts = new int[labels.length];
            double[] clusterDistSums = new double[labels.length];

            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                double distance = euclideanDistance(point, points[j]);
                if (labels[j] == cluster) {
                    intraClusterDist += distance;
                    intraClusterCount++;
                } else {
                    clusterDistSums[labels[j]] += distance;
                    clusterCounts[labels[j]]++;
                }
            }

            a[i] = (intraClusterCount == 0) ? 0 : intraClusterDist / intraClusterCount;

            for (int k = 0; k < clusterCounts.length; k++) {
                if (k != cluster && clusterCounts[k] > 0) {
                    double avgDist = clusterDistSums[k] / clusterCounts[k];
                    if (avgDist < nearestClusterDist) {
                        nearestClusterDist = avgDist;
                    }
                }
            }

            b[i] = nearestClusterDist;
        }

        double silhouetteSum = 0.0;
        for (int i = 0; i < n; i++) {
            silhouetteSum += (b[i] - a[i]) / Math.max(a[i], b[i]);
        }

        return silhouetteSum / n;
    }

    public static double calculateNMI(int[] trueLabels, int[] predictedLabels) {
        int n = trueLabels.length;
        Map<Integer, Integer> trueLabelCount = new HashMap<>();
        Map<Integer, Integer> predictedLabelCount = new HashMap<>();
        Map<Integer, Map<Integer, Integer>> contingencyTable = new HashMap<>();

        for (int i = 0; i < n; i++) {
            trueLabelCount.put(trueLabels[i], trueLabelCount.getOrDefault(trueLabels[i], 0) + 1);
            predictedLabelCount.put(predictedLabels[i], predictedLabelCount.getOrDefault(predictedLabels[i], 0) + 1);

            contingencyTable.computeIfAbsent(trueLabels[i], k -> new HashMap<>());
            contingencyTable.get(trueLabels[i]).put(predictedLabels[i], contingencyTable.get(trueLabels[i]).getOrDefault(predictedLabels[i], 0) + 1);
        }

        double mutualInformation = 0.0;
        for (Map.Entry<Integer, Map<Integer, Integer>> entry : contingencyTable.entrySet()) {
            int trueLabel = entry.getKey();
            for (Map.Entry<Integer, Integer> subEntry : entry.getValue().entrySet()) {
                int predictedLabel = subEntry.getKey();
                int count = subEntry.getValue();

                double jointProbability = (double) count / n;
                double trueLabelProbability = (double) trueLabelCount.get(trueLabel) / n;
                double predictedLabelProbability = (double) predictedLabelCount.get(predictedLabel) / n;

                mutualInformation += jointProbability * Math.log(jointProbability / (trueLabelProbability * predictedLabelProbability));
            }
        }

        double trueEntropy = 0.0;
        for (int count : trueLabelCount.values()) {
            double probability = (double) count / n;
            trueEntropy -= probability * Math.log(probability);
        }

        double predictedEntropy = 0.0;
        for (int count : predictedLabelCount.values()) {
            double probability = (double) count / n;
            predictedEntropy -= probability * Math.log(probability);
        }

        return 2 * mutualInformation / (trueEntropy + predictedEntropy);
    }

    private static double euclideanDistance(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            sum += Math.pow(point1[i] - point2[i], 2);
        }
        return Math.sqrt(sum);
    }
}
