import numpy as np
class K_mean:
    """
    Implementation of EM algorithm for unsupervised learning.

    Input:
    -cluster: number of clusters
    -X: training data, with shape (N, d)
    -query: query data, with shape (N, d)

    """

    def __init__(self, cluster):
        self.cluster = cluster
        self.u = None

    def fit(self, X):

        k = self.cluster
        N_samples, N_features = X.shape

        # initialize cluster center
        u = np.random.rand(k, N_features)
        xmin = np.min(X, axis=0)
        xmax = np.max(X, axis=0)
        u = (xmax-xmin)*u+xmin

        # initialize cluster label
        C = np.zeros((N_samples,))

        err = 10000
        tol = 1e-6
        iterations = 0

        # use d to store distances of N data cluster centers
        d = np.zeros((N_samples, k))

        while err>tol and iterations<10000:
            u_old = u.copy()

            for i in np.arange(k):
                d[:,i] = np.linalg.norm((X-u[i]), axis=1)

            # update C
            C = np.argmin(d, axis=1)

            # update u
            for i in np.arange(k):
                u[i] = np.sum(X[C==i], axis=0)/np.sum(C==i)

            err = np.max(np.abs(u-u_old))

            iterations += 1

        self.u = u

    def predict(self, query):

        k = self.cluster
        N_samples, _ = query.shape
        d = np.zeros((N_samples, k))
        for i in np.arange(k):
            d[:,i] = np.linalg.norm((X-u[i]), axis=1)

        pred = np.argmin(d, axis=1)

        return pred
"""
#Usage
model = K_mean(3)
model.fit(X)
pred = model.predict(X)
"""