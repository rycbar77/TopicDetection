import numpy as np
import torch


class KMeans(object):
    def __init__(self,
                 n_clusters,
                 tol=1e-4,
                 max_iter=300,
                 random_state=0,
                 device=torch.device('cpu')):
        self.cluster_centers = []
        self.device = device
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.choice_cluster = None
        self.initial_state = None
        self.random_state = random_state
        self.mSimilar = None

    def get_distance(self, initial_state):
        dis = self.mSimilar[initial_state]
        dis = dis.T
        return torch.from_numpy(dis).to(self.device)

    def fit_predict(self, X, mSimilar, X_array=None, X_torch=None):
        self.mSimilar = mSimilar
        if X_array is None:
            X_array = np.array([[i[1] for i in doc] for doc in X])
        if X_torch is None:
            X_torch = torch.from_numpy(X_array).to(self.device)
        num_samples = X.corpus.corpus.num_docs
        np.random.seed(self.random_state)
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
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
                             in range(len(initial_state))]
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
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
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

    def predict(self, mSimilar):
        tmp = self.mSimilar
        self.mSimilar = mSimilar
        choice_points = [[(j, np.array(self.initial_state[i]).tolist()[j]) for j in range(self.initial_state.shape[1])]
                         for i in range(len(self.initial_state))]
        dis = self.get_distance(choice_points)
        choice_cluster = torch.argmax(dis, dim=1)
        self.mSimilar = tmp
        return choice_cluster.cpu()

# if __name__ == "__main__":
#     x = np.random.randn(1000, 2) / 6
#     x = torch.from_numpy(x)
#     indices = np.random.choice(len(x), 3, replace=False)
#     initial_state = x[indices]
#     dis = pairwise_euclidean(x, initial_state)
#     print(dis)
