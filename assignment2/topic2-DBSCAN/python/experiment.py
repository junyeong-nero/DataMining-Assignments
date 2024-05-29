from DBSCAN import Database, DBSCAN, Utils, Estimate, Evaluation, VDBSCAN    
import matplotlib.pyplot as plt
    
def run(DB, eps, minpts):
    D = DBSCAN(DB, epsilon=eps, minpts=minpts)
    clusters, core_points, border_points, noise_points = D.run()
    
    print('Number of clusters :', len(clusters))
    print('Number of noise :', len(noise_points))
    
    Utils.vis(DB, clusters, border_points, noise_points)
    return clusters
    

### estimate epsilon

def estimate_eps_0(DB, minpts):    
    E = Estimate(DB)
    norm_eps, eps = E.k_dist_epsilon_estimate()
    return norm_eps, eps

def estimate_eps_1(DB, minpts):    
    E = Estimate(DB)
    norm_eps, eps = E.k_dist_farthest()
    return norm_eps, eps

def estimate_eps_2(DB, minpts):    
    E = Estimate(DB)
    norm_eps, eps = E.k_dist_slope()
    return norm_eps, eps

def estimate_eps_3(DB, minpts):
    E = Estimate(DB)
    norm_eps, eps = E.k_average_dist_epsilon_estimate()
    return norm_eps, eps

def estimate_eps_4(DB, minpts):
    E = Estimate(DB)
    norm_eps, eps = E.k_all_dist_epsilon_estimate()
    return norm_eps, eps

    
### estimate minpts    
    
def estimate_minpts_1(DB, epsilon):    
    E = Estimate(DB)
    minpts = E.rule_of_thumbs_minpts_estimate(epsilon)
    return minpts

def estimate_minpts_2(DB, epsilon):    
    E = Estimate(DB)
    minpts = E.average_minpts_estimate(epsilon)
    return minpts
    
def estimate_minpts_3(DB, epsilon):    
    E = Estimate(DB)
    minpts = E.silhouette_minpts_estimate(epsilon)
    return minpts

def ESTIMATE_EPS(MINPTS):
    
    print('# ESTIMATE EPS with MINPTS =', MINPTS)
    
    plt.figure(figsize=(5, 5))
    
    estimated_eps = [
        estimate_eps_0(DB, MINPTS),
        estimate_eps_1(DB, MINPTS),
        estimate_eps_2(DB, MINPTS),
        estimate_eps_3(DB, MINPTS),
        estimate_eps_4(DB, MINPTS),
    ]
    
    plt.legend()
    plt.show()
    
    print(estimated_eps)
    
    for index, result in enumerate(estimated_eps):
        
        norm_eps, eps = result
        print('### ESTIMATE EPS : #', index)
        print('norm_eps : ', norm_eps)
        print('eps : ', eps)
        
        clusters = run(DB, eps, MINPTS)
        
        # evaluation
        silhouette_score = Evaluation.silhouette_score(DB, clusters)
        NMI = Evaluation.NMI(DB, clusters, cluster_ground_truth)
        print('silhouette_score : ', silhouette_score)
        print('NMI : ', NMI)
        print()
        
        
def ESTIMATE_MINPTS(EPS):
    
    print('# ESTIMATE MINPTS with EPS =', EPS)
    estimated_minpts = [
        estimate_minpts_1(DB, EPS),
        estimate_minpts_2(DB, EPS),
        estimate_minpts_3(DB, EPS)
    ]
    
    for index, minpts in enumerate(estimated_minpts):
        
        print('### ESTIMATE MINPTS : #', index)
        print('minpts : ', minpts)
        clusters = run(DB, EPS, minpts)
        
        # evaluation
        silhouette_score = Evaluation.silhouette_score(DB, clusters)
        NMI = Evaluation.NMI(DB, clusters, cluster_ground_truth)
        print('silhouette_score : ', silhouette_score)
        print('NMI : ', NMI)
        print()


def ground_truth():
    print('# GROUND TRUTH')
    print('Number of clusters :', len(cluster_ground_truth))
    Utils.vis(DB, cluster_ground_truth)
    

def EXPERIMENT_VDBSCAN():
    
    plt.figure(figsize=(5, 5))
    
    E = Estimate(DB)
    norm_eps, eps = E.k_dist_multiple_epsilon_estimate(number_of_density=2)
    print(eps)

    plt.legend()
    plt.show()
    
    D = VDBSCAN(DB, epsilons=eps[::-1], minpts=4)
    clusters, core_points, border_points, noise_points = D.run()
    
    print('Number of clusters :', len(clusters))
    print('Number of noise :', len(noise_points))
    
    silhouette_score = Evaluation.silhouette_score(DB, clusters)
    NMI = Evaluation.NMI(DB, clusters, cluster_ground_truth)
    print('silhouette_score : ', silhouette_score)
    print('NMI : ', NMI)
    print()
    
    Utils.vis(DB, clusters, border_points, noise_points)


import sys

if __name__ == '__main__':
    
    args = sys.argv
    
    DATA = ['artd-31.csv',
            'artset1.csv',
            'square4.csv',
            'elly-2d10c13s.csv',
            'donut-1.csv',
            'ring.csv']
    
    DB = None
    if len(args) == 1:
        DB = Database().read_DB(DATA[1])
    if len(args) == 2:
        DB = Database().read_DB(args[1])
    cluster_ground_truth = Utils.generate_ground_truth(DB)
    
    ground_truth()
    
    # ESTIMATE_EPS(4)
    ESTIMATE_MINPTS(0.02)
    
    EXPERIMENT_VDBSCAN()
    