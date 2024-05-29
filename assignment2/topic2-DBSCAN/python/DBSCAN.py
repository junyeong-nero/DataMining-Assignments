import csv
import random
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Data Structure for Union-find
class DisjointSet:

    def __init__(self, vertices):
        self.parent = {vertex: vertex for vertex in vertices} 
        self.rank = {vertex: 0 for vertex in vertices}

    def find(self, x):
        if self.parent[x] != x:
            return self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        x, y = self.find(a), self.find(b)
        if x != y:
            if self.rank[x] < self.rank[y]:
                self.parent[x] = y
            else:
                self.parent[y] = x
                self.rank[x] += 1
                         

class Utils:
    
    def generate_ground_truth(DB):
        ground_truth = {}
        for key, point in DB.items():
            cluster_id = point['cluster']
            if cluster_id not in ground_truth:
                ground_truth[cluster_id] = []
            ground_truth[cluster_id].append(key)
            
        return ground_truth
    
    # calculate L_k distance between `point1`, `point2`
    def dist(point1, point2, k=2):
        coord1, coord2 = point1['coord'], point2['coord'], 
        assert len(coord1) == len(coord2)
        
        dim, dis = len(coord1), 0
        for i in range(dim):
            dis += abs(coord1[i] - coord2[i]) ** k
        return dis ** (1 / k)
    
    def min_max_normalize(arr):
        min_value, max_value = min(arr), max(arr)
        return [(x - min_value) / (max_value - min_value) for x in arr]
    
    def mean(arr):
        return sum(arr) / len(arr) if len(arr) > 0 else 0
    
    # Visualization of clustered result
    def vis(DB, clusters, border_points=None, noise_points=None):
    
        plt.figure(figsize=(5, 5))
        
        def plot_points(indices, color):
            x = [DB[index]['coord'][0] for index in indices]
            y = [DB[index]['coord'][1] for index in indices]
            plt.scatter(x, y, color=color, s=5)
        
        colors = ['powderblue', 'midnightblue', 'fuchsia', 'indianred', 'saddlebrown', 'crimson', 'magenta', 'olivedrab', 'yellowgreen', 'gainsboro', 'navy', 'aqua', 'darkslategrey', 'darkslategray', 'red', 'darkturquoise', 'tomato', 'firebrick', 'steelblue', 'orangered', 'olive', 'teal', 'chocolate', 'darkblue', 'dodgerblue', 'forestgreen', 'dimgray', 'mediumvioletred', 'limegreen', 'indigo', 'darkgreen', 'peru', 'skyblue', 'black', 'rosybrown', 'hotpink', 'darksalmon', 'gold', 'maroon']
        color_index = 0
        
        for id, cluster in clusters.items():
            plot_points(cluster, color=colors[color_index % len(colors)])
            color_index += 1
            
        if border_points:
            plot_points(border_points, color='b')
        if noise_points:
            plot_points(noise_points, color='r')

        plt.show()


class Database:
    
    # read csv file
    def read_csv_file(self, file_path):
        data = []
        with open(file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                data.append(row)
        return data
    
    # convert csv file to dictionary format
    def read_DB(self, path):
        data = self.read_csv_file(path)
        
        db = {}
        
        X, Y = [], []
        for row in data:
            X.append(float(row[1]))
            Y.append(float(row[2]))
            
        X = Utils.min_max_normalize(X)
        Y = Utils.min_max_normalize(Y)
        
        for i, row in enumerate(data):
            index, x, y, origin_cluster = row
            db[index] = {}
            db[index]['coord'] = [X[i], Y[i]]
            db[index]['cluster'] = origin_cluster
        
        return db
    
    # generate random DB
    def generate_DB(self, count=10, size=10):
        db = {}
        for i in range(count):
            index = 'p' + str(i)
            db[index] = {}
            db[index]['coord'] = [random.randint(1, size), random.randint(1, size)]
            db[index]['cluster'] = 0
        return db


class DBSCAN:

    def __init__(self, DB, epsilon, minpts, 
                 mask={}) -> None:
        self.epsilon = epsilon
        self.minpts = minpts
        self.DB = DB
        if mask:
            self.DB = {name: self.DB[name] for name in self.DB if name in mask}

    
    # return distances and number of point in epsilon distance
    def _count_point(self, target):
        distances_points, number_points = set(), 0
        for key, point in self.DB.items():
            d = Utils.dist(point, target)
            if d <= self.epsilon:
                distances_points.add((key, d))
                number_points += 1
            
        return distances_points, number_points
        
    # return indices of core points in DB
    def _get_core_points(self):
        indices = set()
        for key, point in self.DB.items():
            dist_points, number_points = self._count_point(point)
            if number_points >= self.minpts:
                indices.add(key)
                
        return indices

    # return indices of border points in DB
    def _get_border_points(self, core_points):
        indices = set()
        for key, point in self.DB.items():
            if key in core_points:
                continue
            dist_points, number_points = self._count_point(point)
            
            # In epsilon boundary, if there is core point => border point
            distance_indices = set([point[0] for point in dist_points])
            if distance_indices & core_points:
                indices.add(key)
                
        return indices
    
    # return indicies of noise points in DB
    def _get_noise_points(self, core_points, border_points):
        return set(self.DB.keys()) - core_points - border_points
    
    # clustering with core points
    def _cluster_from_core_points(self, core_points, start_index=1):
        D = DisjointSet(core_points)
        
        for i in core_points:
            for j in core_points:
                if i == j or D.find(i) == D.find(j):
                    continue
                if Utils.dist(self.DB[i], self.DB[j]) <= self.epsilon:
                    D.union(i, j)
        
        index = start_index
        cluster_ids, clusters = {}, {}
        for i in core_points:
            id = D.find(i)
            if id not in cluster_ids:
                cluster_ids[id] = index
                index += 1
            
            new_id = cluster_ids[id]
            if new_id not in clusters:
                clusters[new_id] = []
            clusters[new_id].append(i)
        
        return clusters
    
    def run(self, start_index=1):
        
        core_points = self._get_core_points()
        # print(core_points)
        
        border_points = self._get_border_points(core_points)
        # print(border_points)
        
        noise_points = self._get_noise_points(core_points, border_points)
        # print(noise_points)
        
        clusters = self._cluster_from_core_points(core_points, start_index=start_index)
        return clusters, core_points, border_points, noise_points


class VDBSCAN:
    
    def __init__(self, DB, epsilons, minpts) -> None:
        self.DB = DB
        self.epsilons = epsilons
        self.minpts = minpts
        
    def run(self):
        all_clusters = {}
        start_index = 1
        for epsilon in self.epsilons:
            print('DBSCAN with eps =', epsilon)
            D = DBSCAN(self.DB, epsilon=epsilon, minpts=self.minpts)
            clusters, core_pts, border_pts, noise_pts = D.run(start_index=start_index)
            
            start_index += len(clusters)
            all_clusters.update(clusters)
            
            # update DB for noise_pts
            self.DB = {key: self.DB[key] for key in self.DB.keys() if key in noise_pts}
            
        return all_clusters, core_pts, border_pts, noise_pts
        
            
class Estimate:
    
    def __init__(self, DB) -> None:
        self.DB = DB
    
    ### MINPTS Estimation
    
    # 1. Rule of thumbs
    def rule_of_thumbs_minpts_estimate(self, eps):
        for point in self.DB.values():
            D = len(point['coord'])
            break
        return D + 1
    
    # 2. Average number of point in epsilon boundary
    def average_minpts_estimate(self, eps):
        count = 0
        for target in self.DB.values():
            for point in self.DB.values():
                if target == point:
                    continue
                d = Utils.dist(point, target)
                if d <= eps:
                    count += 1
                    
        return count / len(self.DB)
    
    # 3. silhouette brute-force
    def silhouette_minpts_estimate(self, eps):
        
        highest_score = -1
        highest_score_minpts = 0
        
        for test_minpts in range(3, 10):
            D = DBSCAN(self.DB, epsilon=eps, minpts=test_minpts)
            clusters, core_points, border_points, noise_points = D.run()
            
            silhouette_score = Evaluation.silhouette_score(self.DB, clusters=clusters)
            print('[silhouette] minpts :', test_minpts, 'silhouette_score :', silhouette_score)
            
            if silhouette_score > highest_score:
                highest_score = silhouette_score
                highest_score_minpts = test_minpts
            else:
                break
        
        return highest_score_minpts
            
    
    ### EPS Estimation
    
    # 1. fartest point with y = x
    def k_dist_farthest(self, k=4, vis=True):
        norm_knee, knee = self.fartestKnee(k=k, vis=vis)
        return norm_knee, knee
    
    # 2. slope AE-DBSCAN with k-dist plot
    def k_dist_slope(self, k=4, vis=True):
        norm_knee, knee = self.slopeKnee(k=k, vis=vis)
        return norm_knee, knee
    
    # 3. kneedle with k-dist plot
    def k_dist_epsilon_estimate(self, k=4, vis=True):
        norm_knee, knee = self.kneedle(k=k, vis=vis)
        return norm_knee[0], knee[0]

    # 4. kneedle in average k-dist plot
    def k_average_dist_epsilon_estimate(self, k=4, vis=True):
        norm_knee, knee = self.kneedle(k=k, plot_type='k-avg-dist', vis=vis)
        return norm_knee[0], knee[0]
    
    # 4. kneedle in average k-dist plot
    def k_all_dist_epsilon_estimate(self, k=4, vis=True):
        norm_knee, knee = self.kneedle(k=k, plot_type='k-all-dist', vis=vis)
        return norm_knee[0], knee[0]
    
    # 5. knee detection in average k-dist plot
    def k_min_dist_epsilon_estimate(self, k=4, vis=True):
        norm_knee, knee = self.kneedle(k=k, plot_type='min', vis=vis)
        return norm_knee[0], knee[0]
        
    # 5. maximum distance of kNN : KNN-DBSCAN
    def maximum_dist_kNN(self, k=3, vis=True):
        norm_knee = 0
        knee, point = self._all_k_maximum_dist(k=k)
        
        if vis:
            x = [point['coord'][0] for point in self.DB.values()]
            y = [point['coord'][1] for point in self.DB.values()]
            
            plt.figure(figsize=(5, 5))
            plt.scatter(x, y, c='b', s=3)
            plt.scatter(point['coord'][0], point['coord'][1], c='r', s=3)
            plt.show()
        
        return norm_knee, knee
    
    def k_dist_multiple_epsilon_estimate(self, number_of_density=2, k=4, vis=True):
        norm_knee, knee = self.kneedle(k=k, S=2, vis=vis)
        cut_norm_knee, cut_knee = [], []
        diff = len(norm_knee) // (number_of_density - 1) - 1
        for index in range(number_of_density):
            cut_norm_knee.append(norm_knee[diff * index])
            cut_knee.append(knee[diff * index])
        
        plt.plot([0, 1], [cut_norm_knee[-1], cut_norm_knee[-1]], linestyle='dotted', label='Minimum Eps')
        plt.plot([0, 1], [cut_norm_knee[0], cut_norm_knee[0]], linestyle='dotted', label='Maximum Eps')
        
        return cut_norm_knee, cut_knee
    
    def fartestKnee(self, k=4, S=2, vis=False):
        y = sorted(self._all_kth_dist(k))
        x = list(range(len(y)))
        
        N = len(y)
        x_N = Utils.min_max_normalize(x)
        y_N = Utils.min_max_normalize(y)
        
        knee_point = 0
        dist = 0
        for i in range(N):
            temp = x_N[i] - y_N[i]
            if temp > dist:
                dist = temp
                knee_point = i
        
    
        colors = list(mcolors.TABLEAU_COLORS.keys())
        if vis:
            plt.plot([0, 1], [y_N[knee_point], y_N[knee_point]], linestyle='dotted', c='r', label='farthest')
        return y_N[knee_point], y[knee_point]
    
    def slopeKnee(self, k=4, vis=False):
        y = sorted(self._all_kth_dist(k))
        x = list(range(len(y)))
        
        N = len(y)
        x_N = Utils.min_max_normalize(x)
        y_N = Utils.min_max_normalize(y)
        
        slopes = []
        for i in range(N - 1):
            slope = (y_N[i + 1] - y_N[i]) / (x_N[i + 1] - x_N[i])
            if slope > 0:
                slopes.append((slope, i))
                
        arr = [x[0] for x in slopes]
        mean = sum(arr) / len(arr)
        std = math.sqrt(sum([(x - mean) ** 2 for x in arr]) / len(arr))
        
        knee_points = []
        for slope in slopes:
            if slope[0] >= mean + std:
                knee_points.append(slope[1])
            
        knee_point = knee_points[0]
        if vis:
            plt.plot([0, 1], [y_N[knee_point], y_N[knee_point]], linestyle='dotted', c='k', label='slope')    
        return y_N[knee_point], y[knee_point]
    
    def kneedle(self, k=4, S=2, plot_type='k-dist', vis=False):
        
        index = 0
        if plot_type == 'k-dist':
            index = 0
            y = sorted(self._all_kth_dist(k))
        if plot_type == 'k-avg-dist':
            index = 1
            y = sorted(self._all_k_average_dist(k))
        if plot_type == 'k-all-dist':
            index = 2
            y = sorted(self._all_k_dist(k))
        if plot_type == 'max':
            index = 3
            y = sorted(self._all_k_max_dist(k))
        x = list(range(len(y)))
        
        N = len(y)
        min_value, max_value = y[0], y[-1]
        
        x_O = Utils.min_max_normalize(x)
        y_O = Utils.min_max_normalize(y)
        
        def transform(arr):
            max_value = max(arr)
            arr = [max_value - x for x in arr]
            arr = arr[::-1]    
            return arr
        
        N = len(y_O)
        # our data is convex, therefore need to transform
        x_N = x_O[:]
        y_N = transform(y_O)
        
        x_D = x_N[:]
        y_D = [y_N[i] - x_N[i] for i in range(N)]

        thresholds = [0] * N
        
        # find local minimum and calculate threshold
        for i in range(1, N - 1):
            a, b = y_D[i] - y_D[i - 1], y_D[i] - y_D[i + 1]
            if a > 0 and b > 0:
                thresholds[i] = y_D[i] - S / (N - 1)
        
        knee_points = []
        
        current_threshold_index = -1
        current_threshold = 0
        for i in range(N):
            if thresholds[i] == 0:
                if y_D[i] < current_threshold:
                    knee_points.append(-(current_threshold_index + 1))
                    current_threshold = 0
            else:
                current_threshold = thresholds[i]
                current_threshold_index = i
                 
        norm_knee = [y_O[i] for i in knee_points]
        knee = [y[i] for i in knee_points]
        # norm_knee = sorted(list(set(norm_knee)))
        # knee = [elem * (max_value - min_value) + min_value for elem in norm_knee]
        
        colors = list(mcolors.TABLEAU_COLORS.keys())
        if vis:
            plt.plot(x_O, y_O, c=colors[index], label=plot_type)
            # plt.plot([0, 1], [y_O[knee_points[0]], y_O[knee_points[0]]], linestyle='dotted')
        
        return norm_knee, knee
        
    def _k_dist(self, target, k):
        distances = []
        for point in self.DB.values():
            if point == target:
                continue
            d = Utils.dist(point, target)
            distances.append(d)
        distances.sort()
        return distances[:k] if len(distances) >= k else distances
    
    def _all_kth_dist(self, k=4):
        k_dist = []
        for point in self.DB.values():
            k_dist.append(self._k_dist(point, k)[-1])
        return k_dist
    
    def _all_k_dist(self, k=4):
        k_dist = []
        for point in self.DB.values():
            k_dist += self._k_dist(point, k)
        return k_dist
    
    def _all_k_average_dist(self, k=4):
        average_dist = []
        for point in self.DB.values():
            arr = self._k_dist(point, k)
            average_dist += [sum(arr) / len(arr)]
        return average_dist
    
    def _all_k_min_dist(self, k=4):
        average_dist = []
        for point in self.DB.values():
            arr = self._k_dist(point, k)
            average_dist += [min(arr)]
        return average_dist
    
    def _all_k_max_dist(self, k=4):
        average_dist = []
        for point in self.DB.values():
            arr = self._k_dist(point, k)
            average_dist += [max(arr)]
        return average_dist
    
    def _all_k_maximum_dist(self, k=3):
        max_k_dist, max_k_point = 0, None
        for point in self.DB.values():
            # max_k_dist = max(max_k_dist, self._k_dist(point, k)[-1])
            k_th_dist = self._k_dist(point, k)[-1]
            if k_th_dist > max_k_dist:
                max_k_dist = k_th_dist
                max_k_point = point
                
        return max_k_dist, max_k_point
    
    
class Evaluation:
    
    def calculate_intra_cluster_distance(DB, point, cluster):
        distances = [Utils.dist(point, DB[x]) for x in cluster]
        return Utils.mean(distances)
    
    def calculate_neareste_cluster_distance(DB, point, clusters, current_cluster_id):
        avg_distance_from_nearsest_cluster = float('inf')
        for cluster_id, cluster in clusters.items():
            if cluster_id == current_cluster_id:
                continue
            avg_distance = Evaluation.calculate_intra_cluster_distance(DB, point, cluster)
            avg_distance_from_nearsest_cluster = min(avg_distance_from_nearsest_cluster, avg_distance)
        return avg_distance_from_nearsest_cluster

    def silhouette_score(DB, clusters):
        index_to_cluster_id = {}
        for cluster_id, cluster in clusters.items():
            for index in cluster:
                index_to_cluster_id[index] = cluster_id
        # print(index_to_cluster_id)
        
        silhouette_scores = []
        for index, point in DB.items():
            if index not in index_to_cluster_id: # noise point
                continue 
            cluster_id = index_to_cluster_id[index]
            a = Evaluation.calculate_intra_cluster_distance(DB, point, clusters[cluster_id])
            b = Evaluation.calculate_neareste_cluster_distance(DB, point, clusters, cluster_id)
            silhouette_scores.append((b - a) / max(a, b))
            
        return Utils.mean(silhouette_scores)
    
    
    def entropy(clusters):
        N = sum([len(cluster) for cluster in clusters.values()])
        probability = [len(cluster) / N for cluster in clusters.values()]
        return -sum([x * math.log(x) for x in probability if x > 0])
    
    def mutual_information(pred_clusters, origin_clusters):
        index_to_cluster_id = {}
        for cluster_id, cluster in origin_clusters.items():
            for index in cluster:
                index_to_cluster_id[index] = cluster_id
                
        MI = 0
        total_points_in_pred_clusters = sum([len(cluster) for cluster in pred_clusters.values()])
        for cluster_id, cluster in pred_clusters.items():
            count_cluster_id = {}
            for index in cluster:
                if index not in index_to_cluster_id:
                    continue
                count_cluster_id[index_to_cluster_id[index]] = count_cluster_id.get(index_to_cluster_id[index], 0) + 1
            
            probability = []
            N = sum(count_cluster_id.values())
            for count in count_cluster_id.values():
                probability.append(count / N)
                
            entropy = sum([x * math.log(x) for x in probability if x > 0])
            MI += -len(cluster) / total_points_in_pred_clusters * entropy
            
        return MI
        
        
    def NMI(DB, pred_clusters, origin_clusters):
        # NMI = 2 * I(Y; C) / (H(Y) + H(C))
        H_Y = Evaluation.entropy(origin_clusters)
        H_C = Evaluation.entropy(pred_clusters)
        I = H_Y - Evaluation.mutual_information(pred_clusters, origin_clusters)
        # print(I, H_Y, H_C)
        return I / (H_Y + H_C + 1e-10)

