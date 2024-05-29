import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class A2_G5_t1 {
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
            k = estimateK(points);
            System.out.println("Estimated k: " + k);
        }

        A2_G5_t1 kmeans = new A2_G5_t1();
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

    public static int estimateK(double[][] points) {
        int maxK = 50;
        double bestSilhouette = -1;
        int bestK = 2;

        for (int k = 2; k <= maxK; k++) {
            A2_G5_t1 kmeans = new A2_G5_t1();
            List<double[]> centroids = kmeans.kMeansPlusPlus(points, k);
            kmeans.kMeans(points, centroids, 100);
            int[] labels = kmeans.getLabels(points, centroids);
            double silhouetteScore = calculateSilhouetteScore(points, labels);

            if (silhouetteScore > bestSilhouette) {
                bestSilhouette = silhouetteScore;
                bestK = k;
            }
        }

        return bestK;
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

    private static double distanceToClosestCentroid(double[] point, List<double[]> centroids) {
        double minDistance = Double.MAX_VALUE;
        for (double[] centroid : centroids) {
            double distance = euclideanDistance(point, centroid);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        return minDistance;
    }

    private static double euclideanDistance(double[] point1, double[] point2) {
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





