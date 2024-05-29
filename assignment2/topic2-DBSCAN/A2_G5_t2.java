import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

class Point {

    public List<Double> coord;
    public String clusterID;

    Point(double x, double y, String clusterID) {
        this.coord = Arrays.asList(x, y);
        this.clusterID = clusterID;
    }
}

class DisjointSet {
    private Map<String, String> parent;
    private Map<String, Integer> rank;

    public DisjointSet(Set<String> vertices) {
        parent = new HashMap<>();
        rank = new HashMap<>();
        for (String vertex : vertices) {
            parent.put(vertex, vertex);
            rank.put(vertex, 0);
        }
    }

    public String find(String x) {
        if (!parent.get(x).equals(x)) {
            return find(parent.get(x));
        }
        return parent.get(x);
    }

    public void union(String a, String b) {
        String x = find(a);
        String y = find(b);
        if (!x.equals(y)) {
            if (rank.get(x) < rank.get(y)) {
                parent.put(x, y);
            } else {
                parent.put(y, x);
                rank.put(x, rank.get(x) + 1);
            }
        }
    }
}

class Utils {

    public static Map<String, List<String>> generateGroundTruth(Map<String, Point> DB) {
        Map<String, List<String>> result = new HashMap<>();
        for (String pointID : DB.keySet()) {
            Point point = DB.get(pointID);
            if (!result.containsKey(point.clusterID))
                result.put(point.clusterID, new ArrayList<>());
            result.get(point.clusterID).add(pointID);
        } 
        return result;
    }

    public static double dist(Point point1, Point point2, int k) {
        List<Double> coord1 = point1.coord;
        List<Double> coord2 = point2.coord;
        assert coord1.size() == coord2.size();

        int dim = coord1.size();
        double dis = 0;
        for (int i = 0; i < dim; i++) {
            dis += Math.pow(Math.abs(coord1.get(i) - coord2.get(i)), k);
        }
        return Math.pow(dis, 1.0 / k);
    }

    public static double mean(List<Double> arr) {
        if (arr.size() == 0)
            return 0;
        double sum = 0;
        for (double num : arr) {
            sum += num;
        }
        return sum / arr.size();
    }

    public static double min(List<Double> arr) {
        if (arr.size() == 0)
            return 0;
        double minValue = Double.MAX_VALUE;
        for (double num : arr) {
            minValue = Math.min(minValue, num);
        }
        return minValue;
    }

    public static double max(List<Double> arr) {
        if (arr.size() == 0)
            return 0;
        double maxValue = Double.MIN_VALUE;
        for (double num : arr) {
            maxValue = Math.max(maxValue, num);
        }
        return maxValue;
    }

    public static double std(List<Double> arr) {
        if (arr.size() == 0)
            return 0;
        double mean = Utils.mean(arr);
        double sumOfSquares = 0.0;
        for (double num : arr) {
            sumOfSquares += Math.pow(num - mean, 2);
        }
        return Math.sqrt(sumOfSquares / arr.size());
    }

    public static List<Double> normalize(List<Double> arr) {
        double minValue = Double.MAX_VALUE;
        double maxValue = Double.MIN_VALUE;
        for (double num : arr) {
            minValue = Math.min(minValue, num);
            maxValue = Math.max(maxValue, num);
        }
        List<Double> normalizedArr = new ArrayList<>();
        for (int i = 0; i < arr.size(); i++) {
            normalizedArr.add((arr.get(i) - minValue) / (maxValue - minValue));
        }
        return normalizedArr;
    }

    public static boolean isInteger(String str) {
        if (str == null || str.isEmpty()) {
            return false;
        }
        try {
            Integer.parseInt(str);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}

class Database {
    public List<List<String>> readCSVFile(String path) throws IOException {
        List<List<String>> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            List<String> row = new ArrayList<>();
            for (String value : values) {
                row.add(value);
            }
            data.add(row);
        }
        br.close();
        return data;
    }

    public Map<String, Point> readDB(String path) throws IOException {
        List<List<String>> data = readCSVFile(path);

        Map<String, Point> db = new HashMap<>();
        List<Double> X = new ArrayList<>();
        List<Double> Y = new ArrayList<>();
        List<String> clusterID = new ArrayList<>();
        List<String> pointID = new ArrayList<>();

        for (List<String> row : data) {
            pointID.add(row.get(0));
            X.add(Double.parseDouble(row.get(1)));
            Y.add(Double.parseDouble(row.get(2)));
            clusterID.add(row.get(3));
        }

        X = Utils.normalize(X);
        Y = Utils.normalize(Y);

        for (int i = 0; i < X.size(); i++) {
            Point point = new Point(X.get(i), Y.get(i), clusterID.get(i));
            db.put(pointID.get(i), point);
        }

        return db;
    }

    public Map<String, Map<String, Object>> generateDB(int count, int size) {
        Map<String, Map<String, Object>> db = new HashMap<>();
        Random rand = new Random();
        for (int i = 0; i < count; i++) {
            String index = "p" + i;
            double x = rand.nextInt(size) + 1;
            double y = rand.nextInt(size) + 1;
            String cluster = "0";

            Map<String, Object> point = new HashMap<>();
            point.put("coord", List.of(x, y));
            point.put("cluster", cluster);

            db.put(index, point);
        }
        return db;
    }
}

class DBSCAN {
    private double epsilon;
    private int minpts;
    private Map<String, Point> DB;

    public Map<String, List<String>> clusters;
    public Set<String> corePoints, borderPoints, noisePoints;
    public double silhouetteScore, NMI;


    public DBSCAN(Map<String, Point> DB, double epsilon, int minpts) {
        this.epsilon = epsilon;
        this.minpts = minpts;
        this.DB = DB;
    }

    private Set<String> countPoint(Point target) {
        Set<String> distancePoints = new HashSet<>();
        for (Map.Entry<String, Point> entry : DB.entrySet()) {
            String key = entry.getKey();
            Point point = entry.getValue();
            double d = Utils.dist(point, target, 2);
            if (d <= epsilon) {
                distancePoints.add(key);
            }
        }
        return distancePoints;
    }

    public Set<String> getCorePoints() {
        Set<String> indices = new HashSet<>();
        for (Map.Entry<String, Point> entry : DB.entrySet()) {
            String key = entry.getKey();
            Point point = entry.getValue();
            Set<String> distPoints = countPoint(point);
            if (distPoints.size() >= minpts) {
                indices.add(key);
            }
        }
        return indices;
    }

    public Set<String> getBorderPoints(Set<String> corePoints) {
        Set<String> indices = new HashSet<>();
        for (Map.Entry<String, Point> entry : DB.entrySet()) {
            String key = entry.getKey();
            Point point = entry.getValue();
            if (corePoints.contains(key)) {
                continue;
            }

            Set<String> distPoints = countPoint(point);
            distPoints.retainAll(corePoints);
            if (!distPoints.isEmpty()) {
                indices.add(key);
            }
        }
        return indices;
    }

    public Set<String> getNoisePoints(Set<String> corePoints, Set<String> borderPoints) {
        Set<String> noisePoints = new HashSet<>(DB.keySet());
        noisePoints.removeAll(corePoints);
        noisePoints.removeAll(borderPoints);
        return noisePoints;
    }

    public Map<String, List<String>> clusterWithCorePoints(Set<String> corePoints, int startIndex) {
        DisjointSet D = new DisjointSet(corePoints);

        for (String i : corePoints) {
            for (String j : corePoints) {
                if (i == j || D.find(i) == D.find(j)) {
                    continue;
                }
                if (Utils.dist(DB.get(i), DB.get(j), 2) <= epsilon) {
                    D.union(i, j);
                }
            }
        }

        int index = startIndex;
        Map<String, Integer> clusterIds = new HashMap<>();
        Map<String, List<String>> clusters = new HashMap<>();

        for (String i : corePoints) {
            String id = D.find(i);
            if (!clusterIds.containsKey(id)) {
                clusterIds.put(id, index);
                index++;
            }

            String newId = clusterIds.get(id) + "";
            clusters.computeIfAbsent(newId, k -> new ArrayList<>()).add(i);
        }

        return clusters;
    }

    public Map<String, List<String>> run(int startIndex) {
        corePoints = getCorePoints();
        borderPoints = getBorderPoints(corePoints);
        noisePoints = getNoisePoints(corePoints, borderPoints);
        clusters = clusterWithCorePoints(corePoints, startIndex);
        return clusters;
    }

    public double[] eval(Map<String, List<String>> groundTruth) {
        silhouetteScore = Evaluation.silhouetteScore(DB, clusters);
        NMI = Evaluation.NMI(DB, clusters, groundTruth);
        return new double[]{silhouetteScore, NMI};
    }

    public void print() {
        System.out.println("Number of clusters: " + clusters.size());
        System.out.println("Number of noise : " + noisePoints.size());

        int size = clusters.size();
        for (int i = 0; i < size; i++) {
            String clusterID = (i + 1) + "";
            String result = String.join(" ", clusters.get(clusterID));
            System.out.println("Cluster #" + clusterID + " => " + result);
        }
    }
}

class VDBSCAN {
    private Map<String, Point> DB;
    private double[] epsilons;
    private int minpts;

    public Map<String, List<String>> clusters;
    public Set<String> corePoints, borderPoints, noisePoints;

    public VDBSCAN(Map<String, Point> DB, double[] epsilons) {
        this.DB = DB;
        this.epsilons = epsilons;
    }

    public Map<String, List<String>> run() {
        Map<String, List<String>> allClusters = new HashMap<>();
        DBSCAN D = null;

        int startIndex = 1;
        for (double epsilon : epsilons) {

            minpts = 4;
            System.out.println("eps = " + epsilon);
            System.out.println("minpts = " + minpts);

            D = new DBSCAN(DB, epsilon, minpts);
            clusters = D.run(startIndex);

            startIndex += clusters.size();
            allClusters.putAll(clusters);

            corePoints = D.getCorePoints();
            borderPoints = D.getBorderPoints(corePoints);
            noisePoints = D.getNoisePoints(corePoints, borderPoints);

            Map<String, Point> newDB = new HashMap<>();
            for (String key : DB.keySet()) {
                if (noisePoints.contains(key)) {
                    newDB.put(key, DB.get(key));
                }
            }
            DB = newDB;
        }
        return allClusters;
    }

    public void print() {
        System.out.println("Number of clusters: " + clusters.size());
        System.out.println("Number of noise : " + noisePoints.size());
        for (String clusterID : clusters.keySet()) {
            String result = String.join(" ", clusters.get(clusterID));
            System.out.println("Cluster #" + clusterID + " => " + result);
        }
    }

    public double[] eval(Map<String, List<String>> groundTruth) {
        double silhouetteScore = Evaluation.silhouetteScore(DB, clusters);
        double NMI = Evaluation.NMI(DB, clusters, groundTruth);

        System.out.println("Silhouette Score : "  + silhouetteScore);
        System.out.println("NMI : "  + NMI);
        return new double[]{silhouetteScore, NMI};
    }

}

class Estimate {
    private Map<String, Point> DB;
    public Estimate(Map<String, Point> DB) {
        this.DB = DB;
    }

    /* MINPTS ESTIMATION */

    public int minptsRuleOfThumbs(double eps) {
        int D = 0;
        for (Point point : DB.values()) {
            List<Double> coord = point.coord;
            D = coord.size();
            break;
        }
        return 2 * D;
    }

    public int minptsAvgNumberInBoundary(double eps) {
        double count = 0;
        for (Point target : DB.values()) {
            for (Point point : DB.values()) {
                if (target.equals(point))
                    continue;
                double d = Utils.dist(point, target, 2);
                if (d <= eps) {
                    count++;
                }
            }
        }
        return (int) (count / DB.size());
    }

    public int minptsSilhouette(double eps) {
        double maxScore = -1;
        int maxScoreMinpts = 0;

        for (int minpts = 3; minpts < 100; minpts++) {
            DBSCAN D = new DBSCAN(DB, eps, minpts);
            Map<String, List<String>> clusters = D.run(1);

            double score = Evaluation.silhouetteScore(DB, clusters);
            // System.out.println("[silhouette] minpts = " + minpts + " / silhouette_score : " + score);

            if (score > maxScore) {
                maxScore = score;
                maxScoreMinpts = minpts;
            } else {
                break;
            }
        }
        return maxScoreMinpts;
    }    


    /* EPSILON ESTIMATION */

    public double[] epsFarestDistPlot(int minpts) {
        int k = minpts;
        double[] result = kneeFarestPoint(k, "BASIC");
        double normEps = result[0];
        double eps = result[1];
        return new double[]{normEps, eps};
    }

    public double[] epsSlopDistPlot(int minpts) {
        int k = minpts;
        double[] result = kneeSlopePoint(k, "BASIC");
        double normEps = result[0];
        double eps = result[1];
        return new double[]{normEps, eps};
    }

    public double[] epsDistPlot(int minpts) {
        int k = minpts;
        double[][] result = kneedle(k, "BASIC");
        double[] normEps = result[0];
        double[] eps = result[1];
        return new double[]{normEps[0], eps[0]};
    }

    public double[] epsAvgDistPlot(int minpts) {
        int k = minpts;
        double[][] result = kneedle(k, "AVG");
        double[] normEps = result[0];
        double[] eps = result[1];
        return new double[]{normEps[0], eps[0]};
    }

    public double[] epsAllDistPlot(int minpts) {
        int k = minpts;
        double[][] result = kneedle(k, "ALL");
        double[] normEps = result[0];
        double[] eps = result[1];
        return new double[]{normEps[0], eps[0]};
    }

    public double[] epsMinDistPlot(int minpts) {
        int k = minpts;
        double[][] result = kneedle(k, "MIN");
        double[] normEps = result[0];
        double[] eps = result[1];
        return new double[]{normEps[0], eps[0]};
    }

    public double[] epsMaxDistOfkNN(int minpts) {
        double maxKDist = 0;
        int k = minpts;
        for (Point point : DB.values()) {
            double kthDist = kDist(point, k).get(k - 1);
            if (kthDist > maxKDist) {
                maxKDist = kthDist;
            }
        }
        return new double[]{0, maxKDist};
    }

    /* MULTIPLE EPSILON ESTIMATION */

    public double[][] multipleEps(int numberOfDensity, int k) {
        double[][] result = kneedle(k, "AVG");
        double[] normKnee = result[0];
        double[] knee = result[1];

        double[] cutNormKnee = new double[numberOfDensity];
        double[] cutKnee = new double[numberOfDensity];

        int diff = normKnee.length / (numberOfDensity - 1) - 1;
        for (int index = 0; index < numberOfDensity; index++) {
            cutNormKnee[index] = normKnee[diff * index];
            cutKnee[index] = knee[diff * index];
        }

        Arrays.sort(cutNormKnee);
        Arrays.sort(cutKnee);

        return new double[][]{cutNormKnee, cutKnee};
    }


    /* Private Functions */

    private List<Double> transform(List<Double> arr) {
        double max = Collections.max(arr);
        List<Double> transformed = new ArrayList<>();
        for (double value : arr) {
            transformed.add(max - value);
        }
        Collections.reverse(transformed);
        return transformed;
    }

    private double[] kneeSlopePoint(int k, String type) {
        List<Double> y = new ArrayList<>();
        if (type == "BASIC") 
            y = allKthDist(k);
        if (type == "AVG")
            y = allKDistAvg(k);
        if (type == "MIN")
            y = allKDistMin(k);
        Collections.sort(y);

        List<Double> x = new ArrayList<>();
        for (double i = 0; i < y.size(); i++) {
            x.add(i);
        }

        int N = y.size();
        List<Double> normalizedY = Utils.normalize(y);
        List<Double> normalizedX = Utils.normalize(x);
        List<Double> slopes = new ArrayList<>();
        for (int i = 0; i < N - 1; i++) {
            double dx = normalizedX.get(i + 1) - normalizedX.get(i);
            double dy = normalizedY.get(i + 1) - normalizedY.get(i);
            if (dx == 0)
                continue;
            double slope = dy / dx;
            if (slope > 0) {
                slopes.add(slope);
            }
        }

        double mean = Utils.mean(slopes);
        double std = Utils.std(slopes);

        int index = -1;
        for (int i = 0; i < N - 1; i++) {
            double dx = normalizedX.get(i + 1) - normalizedX.get(i);
            double dy = normalizedY.get(i + 1) - normalizedY.get(i);
            if (dx == 0)
                continue;
            double slope = dy / dx;
            if (slope > mean + std) {
                index = i;
                break;
            }
        }

        return new double[]{normalizedY.get(index), y.get(index)};
    }

    private double[] kneeFarestPoint(int k, String type) {
        List<Double> y = new ArrayList<>();
        if (type == "BASIC") 
            y = allKthDist(k);
        if (type == "AVG")
            y = allKDistAvg(k);
        if (type == "MIN")
            y = allKDistMin(k);
        Collections.sort(y);

        List<Double> x = new ArrayList<>();
        for (double i = 0; i < y.size(); i++) {
            x.add(i);
        }

        int N = y.size();
        List<Double> normalizedY = Utils.normalize(y);
        List<Double> normalizedX = Utils.normalize(x);

        int maxIndex = 0;
        double maxValue = 0;
        for (int i = 0; i < N; i++) {
            double temp = normalizedX.get(i) - normalizedY.get(i);
            if (temp > maxValue) {
                maxValue = temp;
                maxIndex = i;
            }
        }

        return new double[]{normalizedY.get(maxIndex), y.get(maxIndex)};
    }

    private double[][] kneedle(int k, String type) {
        List<Double> y = new ArrayList<>();
        if (type == "BASIC") 
            y = allKthDist(k);
        if (type == "AVG")
            y = allKDistAvg(k);
        if (type == "ALL")
            y = allKDist(k);
        if (type == "MIN")
            y = allKDistMin(k);
        if (type == "MAX")
            y = allKDistMax(k);
        Collections.sort(y);

        List<Double> x = new ArrayList<>();
        for (double i = 0; i < y.size(); i++) {
            x.add(i);
        }

        int N = y.size();
        List<Double> normalizedY = Utils.normalize(y);
        List<Double> normalizedX = Utils.normalize(x);

        List<Double> transformedX = normalizedX;
        List<Double> transformedY = transform(normalizedY);

        List<Double> distanceX = new ArrayList<>(transformedX);
        List<Double> distanceY = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            distanceY.add(transformedY.get(i) - transformedX.get(i));
        }

        double[] thresholds = new double[N];
        for (int i = 1; i < N - 1; i++) {
            double a = distanceY.get(i) - distanceY.get(i - 1);
            double b = distanceY.get(i) - distanceY.get(i + 1);
            if (a > 0 && b > 0) {
                thresholds[i] = distanceY.get(i) - 2 / (double)(N - 1);
            }
        }

        List<Integer> kneePoints = new ArrayList<>();
        int currentThresholdIndex = -1;
        double currentThreshold = 0;
        for (int i = 0; i < N; i++) {
            if (thresholds[i] == 0) {
                if (distanceY.get(i) < currentThreshold) {
                    kneePoints.add(N -(currentThresholdIndex + 1));
                    currentThreshold = 0;
                }
            } else {
                currentThreshold = thresholds[i];
                currentThresholdIndex = i;
            }
        }

        double[] normKnee = new double[kneePoints.size()];
        double[] knee = new double[kneePoints.size()];
        for (int i = 0; i < kneePoints.size(); i++) {
            normKnee[i] = normalizedY.get(kneePoints.get(i));
            knee[i] = y.get(kneePoints.get(i));
        }

        return new double[][]{normKnee, knee};
    }

    private List<Double> kDist(Point target, int k) {
        List<Double> distances = new ArrayList<>();
        for (Point point : DB.values()) {
            double d = Utils.dist(point, target, 2);
            if (d > 0)
                distances.add(d);
        }
        distances.sort(null);
        return distances.subList(0, Math.min(k, distances.size()));
    }

    // 
    private List<Double> allKthDist(int k) {
        List<Double> dist = new ArrayList<>();
        for (Point point : DB.values()) {
            List<Double> kDist = kDist(point, k);
            dist.add(kDist.get(kDist.size() - 1));
        }
        return dist;
    }

    private List<Double> allKDist(int k) {
        List<Double> dist = new ArrayList<>();
        for (Point point : DB.values()) {
            List<Double> kDist = kDist(point, k);
            dist.addAll(kDist);
        }
        return dist;
    }

    // SS-DBSCAN used
    private List<Double> allKDistAvg(int k) {
        List<Double> dist = new ArrayList<>();
        for (Point point : DB.values()) {
            List<Double> kDist = kDist(point, k);
            dist.add(Utils.mean(kDist));
        }
        return dist;
    }

    // DMDBSCAN used
    private List<Double> allKDistMin(int k) {
        List<Double> dist = new ArrayList<>();
        for (Point point : DB.values()) {
            List<Double> kDist = kDist(point, k);
            dist.add(Utils.min(kDist));
        }
        return dist;
    }

    // DMDBSCAN used
    private List<Double> allKDistMax(int k) {
        List<Double> dist = new ArrayList<>();
        for (Point point : DB.values()) {
            List<Double> kDist = kDist(point, k);
            dist.add(Utils.max(kDist));
        }
        return dist;
    }
}

class Evaluation {
    public static double calculateIntraClusterDist(Map<String, Point> DB, Point point, List<String> cluster) {
        List<Double> distances = new ArrayList<>();
        for (String index : cluster) {
            Double dist = Utils.dist(DB.get(index), point, 2);
            if (dist > 0)
                distances.add(dist);
        }
        return Utils.mean(distances);
    }

    public static double calculateNearestClusterDist(Map<String, Point> DB, Point point, Map<String, List<String>> clusters, String currentClusterID) {
        double avgDist = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, List<String>> entry : clusters.entrySet()) {
            String clusterID = entry.getKey();
            List<String> cluster = entry.getValue();
            if (clusterID == currentClusterID) {
                continue;
            }
            double dist = calculateIntraClusterDist(DB, point, cluster);
            avgDist = Math.min(avgDist, dist);
        }
        return avgDist;
    }

    public static double silhouetteScore(Map<String, Point> DB, Map<String, List<String>> clusters) {
        Map<String, String> index2clusterID = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : clusters.entrySet()) {
            String clusterID = entry.getKey();
            List<String> cluster = entry.getValue();
            for (String index : cluster) {
                index2clusterID.put(index, clusterID);
            }
        }

        List<Double> score = new ArrayList<>();
        for (Map.Entry<String, Point> entry : DB.entrySet()) {
            String index = entry.getKey();
            Point point = entry.getValue();
            if (!index2clusterID.containsKey(index)) {
                continue;
            }
            String clusterID = index2clusterID.get(index);
            double a = calculateIntraClusterDist(DB, point, clusters.get(clusterID));
            double b = calculateNearestClusterDist(DB, point, clusters, clusterID);
            score.add((b - a) / Math.max(a, b));
        }

        return Utils.mean(score);
    }

    public static double entropy(Map<String, List<String>> clusters) {
        int N = clusters.values().stream().mapToInt(List::size).sum();
        List<Double> probability = new ArrayList<>();
        for (List<String> cluster : clusters.values()) {
            probability.add((double) cluster.size() / N);
        }
        double entropy = 0;
        for (double p : probability) {
            entropy -= p * Math.log(p);
        }
        return entropy;
    }

    public static double mutualInformation(Map<String, List<String>> predClusters, Map<String, List<String>> originClusters) {
        Map<String, String> indexToClusterId = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : originClusters.entrySet()) {
            String clusterId = entry.getKey();
            List<String> cluster = entry.getValue();
            for (String index : cluster) {
                indexToClusterId.put(index, clusterId);
            }
        }

        double MI = 0;
        int totalPointsInPredClusters = predClusters.values().stream().mapToInt(List::size).sum();

        for (Map.Entry<String, List<String>> entry : predClusters.entrySet()) {
            List<String> cluster = entry.getValue();
            Map<String, Integer> countClusterId = new HashMap<>();

            for (String index : cluster) {
                if (!indexToClusterId.containsKey(index)) {
                    continue;
                }
                String clusterId = indexToClusterId.get(index);
                countClusterId.put(clusterId, countClusterId.getOrDefault(clusterId, 0) + 1);
            }

            List<Double> probability = new ArrayList<>();
            int N = countClusterId.values().stream().mapToInt(Integer::intValue).sum();

            for (int count : countClusterId.values()) {
                probability.add((double) count / N);
            }

            double entropy = 0;
            for (double p : probability) {
                if (p > 0) {
                    entropy += p * Math.log(p);
                }
            }

            MI += -((double) cluster.size() / totalPointsInPredClusters) * entropy;
        }

        return MI;
    }

    public static double NMI(Map<String, Point> DB, Map<String, List<String>> predClusters, Map<String, List<String>> originClusters) {
        double H_Y = entropy(originClusters);
        double H_C = entropy(predClusters);
        double I = H_Y - mutualInformation(predClusters, originClusters);
        return I / (H_Y + H_C);
    }
}

class Experiment {

    public static void experimentVBDSCAN(String path) throws IOException {
        Database database = new Database();
        Map<String, Point> DB = database.readDB(path);
        Map<String, List<String>> groundTruth = Utils.generateGroundTruth(DB);

        Estimate E = new Estimate(DB);
        double[] epsilons = E.multipleEps(2, 4)[1];

        VDBSCAN V = new VDBSCAN(DB, epsilons);
        V.run();
        V.print();
        V.eval(groundTruth);
    }

    public static void experiment(String path) throws IOException {
        Database database = new Database();
        Map<String, Point> DB = database.readDB(path);
        Map<String, List<String>> groundTruth = Utils.generateGroundTruth(DB);

        epsilonEstimateExperiment(path, DB, groundTruth, 4);
        minptsEstimateExperiment(path, DB, groundTruth, 0.02);
    }

    public static void epsilonEstimateExperiment(String path, Map<String, Point> DB, Map<String, List<String>> groundTruth, int minpts) {
        Estimate E = new Estimate(DB);

        List<Double> estimatedEpsilons = new ArrayList<>();
        double[] result;

        // 1. kneedle
        result = E.epsDistPlot(minpts);
        estimatedEpsilons.add(result[1]);

        // 2. farthest point
        result = E.epsFarestDistPlot(minpts);
        estimatedEpsilons.add(result[1]);

        // 3. slope based
        result = E.epsSlopDistPlot(minpts);
        estimatedEpsilons.add(result[1]);

        // 4. k-avg-dist + kneedle
        result = E.epsAvgDistPlot(minpts);
        estimatedEpsilons.add(result[1]);

        // 5. k-all-dist + kneedle
        result = E.epsAllDistPlot(minpts);
        estimatedEpsilons.add(result[1]);
        
        for (double eps : estimatedEpsilons) {
            
            DBSCAN D = new DBSCAN(DB, eps, minpts);
            D.run(0);
            D.eval(groundTruth);

            String[] printData = new String[]{
                path, 
                String.format("%.3f", eps * 100) + "",
                D.clusters.size() + "",
                D.noisePoints.size() + "",
                String.format("%.3f", D.silhouetteScore),
                String.format("%.3f", D.NMI),
            };
            System.out.println(String.join(",", printData));
        }
    }

    public static void minptsEstimateExperiment(String path, Map<String, Point> DB, Map<String, List<String>> groundTruth, double epsilon) {
        Estimate E = new Estimate(DB);

        List<Integer> estimatedMinpts = new ArrayList<>();
        int result;

        // 1. rule of thumbs
        result = E.minptsRuleOfThumbs(epsilon);
        estimatedMinpts.add(result);
        
        // 2. silhouette
        result = E.minptsSilhouette(epsilon);
        estimatedMinpts.add(result);

        // 3. avg # of points
        result = E.minptsAvgNumberInBoundary(epsilon);
        estimatedMinpts.add(result);

        for (int minpts : estimatedMinpts) {
            DBSCAN D = new DBSCAN(DB, epsilon, (int) minpts);
            D.run(0);
            D.eval(groundTruth);

            String[] printData = new String[]{
                path,
                minpts + "",
                D.clusters.size() + "",
                D.noisePoints.size() + "",
                String.format("%.3f", D.silhouetteScore),
                String.format("%.3f", D.NMI),
            };
            System.out.println(String.join(",", printData));
        }
    }

}

public class A2_G5_t2 {

    public static void main(String[] args) throws IOException {

        if (args.length == 1) {
            Experiment.experiment(args[0]);
            return;
        }

        Database database = new Database();
        Map<String, Point> DB = database.readDB(args[0]);

        Estimate E = new Estimate(DB);

        double epsilon = 0.02;
        int minpts = 4;

        if (args.length == 2) {
            if (Utils.isInteger(args[1])) {
                minpts = Integer.parseInt(args[1]);
                double[] result = E.epsAvgDistPlot(minpts);
                epsilon = result[1];
                System.out.println("Estimated eps : " + result[1]);
            } else {
                epsilon = Double.parseDouble(args[1]);
                minpts = E.minptsRuleOfThumbs(epsilon);
                System.out.println("Estimated MinPts : " + minpts);
            }
        }

        if (args.length == 3) {
            if (Utils.isInteger(args[1])) {
                epsilon = Double.parseDouble(args[2]);
                minpts = Integer.parseInt(args[1]);
            } else {
                epsilon = Double.parseDouble(args[1]);
                minpts = Integer.parseInt(args[2]);
            }
        }

        DBSCAN dbscan = new DBSCAN(DB, epsilon, minpts);
        dbscan.run(1);
        dbscan.print();

    }
}


