from random import random

import numpy as np
import torch
from gensim_kmeans import KMeans


# TODO: Finish get_cent
class KMeansPlusPlus(KMeans):
    def get_cent(self, points):
        # dummy dummy type convert
        m, n = np.shape(points)
        self.cluster_centers = np.mat(np.zeros((self.n_clusters, n)))
        index = np.random.randint(0, m)
        self.cluster_centers[0] = np.copy(points[index])

        d = [0.0 for _ in range(m)]

        for i in range(1, self.n_clusters):
            dis = self.get_distance(points(self.cluster_centers))
            choice_points = torch.argmax(dis, dim=0)
            sum_all = torch.sum(choice_points)
            sum_all *= random()
            for j, di in enumerate(d):
                sum_all -= di
                if sum_all > 0:
                    continue
                self.cluster_centers[i] = np.copy(points[j])
                break
        return self.cluster_centers
