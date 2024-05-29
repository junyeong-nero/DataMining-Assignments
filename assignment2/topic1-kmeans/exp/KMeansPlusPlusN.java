import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class KMeansPlusPlusN {
    private static final Random random = new Random();

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java KMeansPlusPlus <input_csv_file> [<k>]");
            return;
        }

        String csvFile = args[0];
        Integer k = args.length > 1 ? Integer.parseInt(args[1]) : null;

        double[][] points = readCSV(csvFile);
        if (k == null) {
            int maxK = 50;
            k = estimateK(points, maxK);
            System.out.println("Estimated k: " + k);
        }

        KMeansPlusPlus kmeans = new KMeansPlusPlus();
        List<double[]> centroids = kmeans.kMeansPlusPlus(points, k);

        kmeans.kMeans(points, centroids, 100);

        // 클러스터 결과 출력 및 저장
        int[] labels = kmeans.getLabels(points, centroids);
        String clustersOutput = getClustersOutput(labels, points);
        System.out.println(clustersOutput);
        writeToFile("clusters_output.txt", clustersOutput);
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

    public static int estimateK(double[][] points, int maxK) {
        double[] costs = new double[maxK];

    // 각 k에 대한 비용을 계산합니다.
        for (int k = 1; k <= maxK; k++) {
            KMeansPlusPlus kmeans = new KMeansPlusPlus();
            List<double[]> centroids = kmeans.kMeansPlusPlus(points, k);
            kmeans.kMeans(points, centroids, 100);
            costs[k - 1] = kmeans.calculateCost(points, centroids);
        }

    // 비용 감소율 계산
        double[] reductionRates = new double[maxK - 1];
        for (int k = 1; k < maxK; k++) {
            reductionRates[k - 1] = (costs[k - 1] - costs[k]) / costs[k - 1];
        }

    // 감소율 임계값 설정 (예: 0.1 이하가 되면 최적의 K 값으로 판단)
        double threshold = 0.1;
        int bestK = 1;
        for (int k = 1; k < maxK; k++) {
            if (reductionRates[k - 1] < threshold) {
                bestK = k;
                break;
            }
        }

        return bestK;
    }

    public double calculateCost(double[][] points, List<double[]> centroids) {
        double cost = 0.0;
        for (double[] point : points) {
            cost += distanceToClosestCentroid(point, centroids);
        }
        return cost;
    }

    public List<double[]> kMeansPlusPlus(double[][] points, int k) {
        int n = points.length;
        List<double[]> centroids = new ArrayList<>();
        Set<double[]> uniqueCentroids = new HashSet<>();

        double[] firstCentroid = points[random.nextInt(n)];
        centroids.add(firstCentroid);
        uniqueCentroids.add(firstCentroid);

        for (int i = 1; i < k; i++) {
            double[] distances = new double[n];
            double totalDistance = 0.0;

            for (int j = 0; j < n; j++) {
                distances[j] = distanceToClosestCentroid(points[j], centroids);
                totalDistance += distances[j] * distances[j];
            }

            double randomValue = random.nextDouble() * totalDistance;
            for (int j = 0; j < n; j++) {
                randomValue -= distances[j] * distances[j];
                if (randomValue <= 0) {
                    if (uniqueCentroids.add(points[j])) {
                        centroids.add(points[j]);
                    } else {
                        i--;
                    }
                    break;
                }
            }
        }

        return centroids;
    }

    public void kMeans(double[][] points, List<double[]> centroids, int maxIterations) {
        int n = points.length;
        int k = centroids.size();
        int[] labels = new int[n];

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            boolean changed = false;

            for (int i = 0; i < n; i++) {
                int newLabel = findClosestCentroid(points[i], centroids);
                if (newLabel != labels[i]) {
                    changed = true;
                    labels[i] = newLabel;
                }
            }

            double[][] newCentroids = new double[k][points[0].length];
            int[] counts = new int[k];
            for (int i = 0; i < n; i++) {
                int label = labels[i];
                for (int j = 0; j < points[i].length; j++) {
                    newCentroids[label][j] += points[i][j];
                }
                counts[label]++;
            }
            for (int i = 0; i < k; i++) {
                if (counts[i] > 0) {
                    for (int j = 0; j < newCentroids[i].length; j++) {
                        newCentroids[i][j] /= counts[i];
                    }
                }
            }

            for (int i = 0; i < k; i++) {
                centroids.set(i, newCentroids[i]);
            }

            if (!changed) {
                break;
            }
        }
    }

    public int[] getLabels(double[][] points, List<double[]> centroids) {
        int[] labels = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            labels[i] = findClosestCentroid(points[i], centroids);
        }
        return labels;
    }

    private int findClosestCentroid(double[] point, List<double[]> centroids) {
        int closest = 0;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < centroids.size(); i++) {
            double distance = euclideanDistance(point, centroids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                closest = i;
            }
        }
        return closest;
    }

    private double distanceToClosestCentroid(double[] point, List<double[]> centroids) {
        double minDistance = Double.MAX_VALUE;
        for (double[] centroid : centroids) {
            double distance = euclideanDistance(point, centroid);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        return minDistance;
    }

    private double euclideanDistance(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            sum += Math.pow(point1[i] - point2[i], 2);
        }
        return Math.sqrt(sum);
    }

    private static String getClustersOutput(int[] labels, double[][] points) {
        int k = 0;
        for (int label : labels) {
            if (label + 1 > k) {
                k = label + 1;
            }
        }

        List<List<String>> clusters = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            clusters.add(new ArrayList<>());
        }

        for (int i = 0; i < labels.length; i++) {
            clusters.get(labels[i]).add("p" + (i + 1));
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < clusters.size(); i++) {
            sb.append("Cluster #").append(i + 1).append(" => ").append(String.join(" ", clusters.get(i))).append("\n");
        }
        return sb.toString();
    }

    private static void writeToFile(String fileName, String content) {
        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write(content);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}