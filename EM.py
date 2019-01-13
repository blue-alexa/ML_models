import numpy as np
from scipy.stats import multivariate_normal

class EM:
    """
    Implementation of EM algorithm for unsupervised learning.

    Input:
    -cluster: number of clusters
    -X: training data, with shape (N, d)
    -query: query data, with shape (N, d)

    """
    def __init__(self, cluster):
        self.cluster = cluster
        self.a = None
        self.u = None
        self.S = None

    def fit(self, X):
        C = self.cluster
        N_samples, N_features = X.shape

        # initializations

        # For 𝑘=1, 2, 3,…𝐶, set 𝑎_𝑘=1/𝐶.
        a = np.zeros((C,))
        a.fill(1/C)

        # For 𝑘=1, 2, 3,…𝐶, set 𝝁_𝑘= 𝑑-dimensional random vector whose 𝑗^th component lies between the minimum and maximum of the 𝑗^th components of 𝑋.
        u = np.random.rand(C, N_features)
        xmin = np.min(X, axis=0)
        xmax = np.max(X, axis=0)
        u = (xmax-xmin)*u+xmin

        # For 𝑘=1, 2, 3,…𝐶, set 𝑆_𝑘=Diagonal matrix of size 𝑑×𝑑 with diagonal entries taken to be the variance of the respective columns of 𝑋.
        S = [np.eye(N_features)*np.var(X, axis=0)]*C

        # Initialize matrix of posterior probabilities 𝑃 of size 𝑁×𝐶.
        p = np.zeros((N_samples, C))

        err = 10000
        tol = 1e-6
        iterations = 0

        while err>tol and iterations<10000:
            u_old = u.copy()

            # weighted pdf
            for i in np.arange(C):
                p[:, i] = a[i]*multivariate_normal.pdf(X, mean=u[i], cov=S[i])

            # normalize p
            p = p/np.sum(p, axis=1, keepdims = True)

            # update a
            a = np.sum(p, axis=0)/N_samples

            # update u
            w = p/np.sum(p, axis=0)
            for i in np.arange(C):
                u[i] = np.sum(w[:, i].reshape(-1,1)*X, axis=0)

            # calculate error
            err = np.max(np.abs(u-u_old))

            # update S
            for i in np.arange(C):
                S[i] = np.dot((w[:,i].reshape(-1,1)*(X-u[i])).T, (X-u[i]))

            iterations += 1

        self.a = a
        self.u = u
        self.S = S

    def predict(self, query):

        if self.a is not None:
            N_samples, _ = query.shape
            p = np.zeros((N_samples, self.cluster))

            # weighted pdf
            for i in np.arange(self.cluster):
                p[:, i] = self.a[i]*multivariate_normal.pdf(query, mean=self.u[i], cov=self.S[i])

            # normalize p
            p = p/np.sum(p, axis=1, keepdims = True)

            return p
"""
# Usage:
model = EM(3)
model.fit(X)
p = model.predict(X)
"""