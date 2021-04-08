from functools import partial
import numpy as np

import torch


def pairwise_euclidean(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)

    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0

    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)

    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)

    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


class KMeans(object):
    def __init__(self,
                 n_clusters,
                 distance='euclidean',
                 tol=1e-4,
                 max_iter=300,
                 random_state=0,
                 device=torch.device('cpu')):
        self.cluster_centers = []
        if distance == 'euclidean':
            self.pairwise_distance_function = partial(pairwise_euclidean, device=device)
        elif distance == 'cosine':
            self.pairwise_distance_function = partial(pairwise_cosine, device=device)
        else:
            raise NotImplementedError
        self.device = device
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.choice_cluster = None
        self.initial_state = None
        self.random_state = random_state

    def fit_predict(self, X):
        X = X.float()
        X = X.to(self.device)
        num_samples = len(X)
        np.random.seed(self.random_state)
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        initial_state = X[indices]
        dis = self.pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(self.device)

        iteration = 0
        while True:

            dis = self.pairwise_distance_function(X, initial_state)

            choice_cluster = torch.argmin(dis, dim=1)

            initial_state_pre = initial_state.clone()

            for index in range(self.n_clusters):
                selected = torch.nonzero(choice_cluster == index).squeeze().to(self.device)

                selected = torch.index_select(X, 0, selected)

                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]

                initial_state[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
                ))

            iteration = iteration + 1

            if center_shift ** 2 < self.tol:
                break
            if self.max_iter != 0 and iteration >= self.max_iter:
                break

        return choice_cluster.cpu(), initial_state.cpu()

    def fit(self, X):
        X = X.float()
        X = X.to(self.device)
        num_samples = len(X)
        np.random.seed(self.random_state)
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        self.initial_state = X[indices]
        dis = self.pairwise_distance_function(X, self.initial_state)
        choice_points = torch.argmin(dis, dim=0)
        self.initial_state = X[choice_points]
        self.initial_state = self.initial_state.to(self.device)

        iteration = 0
        while True:
            dis = self.pairwise_distance_function(X, self.initial_state)
            self.choice_cluster = torch.argmin(dis, dim=1)

            initial_state_pre = self.initial_state.clone()

            for index in range(self.n_clusters):
                selected = torch.nonzero(self.choice_cluster == index).squeeze().to(self.device)

                selected = torch.index_select(X, 0, selected)

                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]

                self.initial_state[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.initial_state - initial_state_pre) ** 2, dim=1)
                ))

            iteration = iteration + 1

            if center_shift ** 2 < self.tol:
                break
            if self.max_iter != 0 and iteration >= self.max_iter:
                break
        self.cluster_centers = self.initial_state
        return self

    def predict(self, X):
        X = X.float()
        X = X.to(self.device)

        dis = self.pairwise_distance_function(X, self.initial_state)
        choice_cluster = torch.argmin(dis, dim=1)

        return choice_cluster.cpu()


if __name__ == "__main__":
    x = np.random.randn(1000, 2) / 6
    x = torch.from_numpy(x)
    indices = np.random.choice(len(x), 3, replace=False)
    initial_state = x[indices]
    dis = pairwise_euclidean(x, initial_state)
    print(dis)
