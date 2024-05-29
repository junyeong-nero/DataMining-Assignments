from DBSCAN import Database, Utils, Estimate, Evaluation
from DBSCAN import DBSCAN, VDBSCAN
    
import matplotlib.pyplot as plt

### both eps and minpts are given

def TEST1(EPS, MINPTS):
    D = DBSCAN(DB, EPS, MINPTS)
    clusters, core_points, border_points, noise_points = D.run()
    Utils.vis(DB, clusters, border_points, noise_points)
    
def TEST2(MINPTS):
    E = Estimate(DB)
    norm_eps, eps = E.k_dist_epsilon_estimate()
    print('Estimated Epsilon :', eps)
    
    D = DBSCAN(DB, eps, MINPTS)
    clusters, core_points, border_points, noise_points = D.run()
    Utils.vis(DB, clusters, border_points, noise_points)
    
def TEST3(EPS):
    E = Estimate(DB)
    minpts = E.rule_of_thumbs_minpts_estimate()
    print('Estimated Minpts :', minpts)
    
    D = DBSCAN(DB, EPS, minpts)
    clusters, core_points, border_points, noise_points = D.run()
    Utils.vis(DB, clusters, border_points, noise_points)
    
    
import sys

if __name__ == '__main__':
    
    args = sys.argv
    DB = Database().read_DB(args[1])
    cluster_ground_truth = Utils.generate_ground_truth(DB)
    
    # TEST1(0.02, 4)
    TEST2(4)