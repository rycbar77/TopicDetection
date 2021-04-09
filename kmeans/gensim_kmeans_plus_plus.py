from kmeans.gensim_kmeans import KMeans
import numpy as np
import torch
from random import random, seed


class KMeansPlusPlus(KMeans):
    def get_cent(self, points, k):
        indices = []
        m, n = np.shape(points)
        cluster_centers = []
        np.random.seed(self.random_state)
        index = np.random.randint(0, m)
        indices.append(index)
        cluster_centers.append(points[index])
        seed(self.random_state)
        for i in range(1, k):
            choice_points = [[(j, np.array(cluster_centers[ii]).tolist()[j]) for j in range(n)] for ii in
                             range(i)]

            dis = self.get_distance(choice_points)
            choice_cluster = torch.argmax(dis, dim=1).unsqueeze(0)
            ind = [[i for i in range(m)], choice_cluster]
            dis = dis[ind].squeeze()
            sum_all = torch.sum(dis).item()
            sum_all *= random()
            for j, di in enumerate(dis):
                sum_all -= di.item()
                if sum_all > 0:
                    continue
                cluster_centers.append(points[j])
                indices.append(j)
                break
        return indices

    def fit_predict(self, X, mSimilar, X_array=None, X_torch=None):
        self.mSimilar = mSimilar
        if X_array is None:
            X_array = np.array([[i[1] for i in doc] for doc in X])
        if X_torch is None:
            X_torch = torch.from_numpy(X_array).to(self.device)
        num_samples = X.corpus.corpus.num_docs
        np.random.seed(self.random_state)
        # indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        indices = self.get_cent(X_array, self.n_clusters)
        dis = self.get_distance(X[indices])
        choice_points = np.array(torch.argmax(dis, dim=0))
        initial_state = X_torch[choice_points]
        choice_points = X[choice_points]
        iteration = 0
        while True:
            dis = self.get_distance(choice_points)
            choice_cluster = torch.argmax(dis, dim=1)
            initial_state_pre = initial_state.clone()
            for index in range(self.n_clusters):
                selected = torch.nonzero(torch.from_numpy(np.array(choice_cluster == index))).squeeze().to(self.device)
                selected = torch.index_select(X_torch, 0, selected)
                if selected.shape[0] == 0:
                    selected = X_torch[torch.randint(len(X), (1,))]

                initial_state[index] = selected.mean(dim=0)
            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
                ))
            choice_points = [[(j, np.array(initial_state[i]).tolist()[j]) for j in range(initial_state.shape[1])] for i
                             in
                             range(len(initial_state))]
            iteration = iteration + 1

            if center_shift ** 2 < self.tol:
                break
            if self.max_iter != 0 and iteration >= self.max_iter:
                break
            self.initial_state = initial_state.cpu()
        return choice_cluster.cpu(), initial_state.cpu()

    def fit(self, X, mSimilar, X_array=None, X_torch=None):
        self.mSimilar = mSimilar
        num_samples = X.corpus.corpus.num_docs
        if X_array is None:
            X_array = np.array([[i[1] for i in doc] for doc in X])
        if X_torch is None:
            X_torch = torch.from_numpy(X_array).to(self.device)
        np.random.seed(self.random_state)
        # indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        indices = self.get_cent(X_array, self.n_clusters)
        dis = self.get_distance(X[indices])
        choice_points = np.array(torch.argmax(dis, dim=0))
        self.initial_state = X_torch[choice_points]
        choice_points = X[choice_points]
        iteration = 0
        while True:
            dis = self.get_distance(choice_points)
            choice_cluster = torch.argmax(dis, dim=1)
            initial_state_pre = self.initial_state.clone()
            for index in range(self.n_clusters):
                selected = torch.nonzero(
                    torch.from_numpy(np.array(choice_cluster == index)).to(self.device)).squeeze().to(self.device)
                selected = torch.index_select(X_torch, 0, selected)
                if selected.shape[0] == 0:
                    selected = X_torch[torch.randint(len(X), (1,))]
                self.initial_state[index] = selected.mean(dim=0)
            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.initial_state - initial_state_pre) ** 2, dim=1)
                ))
            choice_points = [
                [(j, np.array(self.initial_state[i]).tolist()[j]) for j in range(self.initial_state.shape[1])] for i in
                range(len(self.initial_state))]
            iteration = iteration + 1

            if center_shift ** 2 < self.tol:
                break
            if self.max_iter != 0 and iteration >= self.max_iter:
                break

        self.cluster_centers = self.initial_state
        return self
