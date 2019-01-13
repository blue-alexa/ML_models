import numpy as np
class perceptron:
    """
    Implementation of perceptron algorithm.

    Input:
    -X: training data, with shape (N, d)
    -query: query data, with shape (N, d)

    """
    def __init__(self, max_iterations=None):
        if max_iterations != None:
            self.iterations = max_iterations
        else:
            self.iterations = 10000
        self.Keslerlabel = None
        self.w = None

    def _Kesler_construction(self, X, y):
        # convert y to Kesler constructions
        class_values = np.unique(y)
        class_num = len(class_values)
        yc = np.zeros_like(y)

        for i, value in enumerate(class_values):
            yc[y==value]=i

        KC = np.zeros((len(X), class_num))
        KC.fill(-1)
        KC[range(len(X)), yc]=1

        self.Keslerlabel = {ind:value for ind, value in enumerate(class_values)}

        return KC

    def _Xa_construction(self, X):
        # add x0=1
        Xa = np.hstack((np.ones((len(X),1)), X))
        return Xa

    def fit(self, X, y):
        Xa = self._Xa_construction(X)
        KC = self._Kesler_construction(X, y)
        k = KC.shape[1]
        N_samples, N_features = Xa.shape
        w = np.ones((N_features, k))
        w[0,:]=0

        iterations = 0
        err = np.prod(Xa.shape)
        best_err = err
        best_w = w

        #Keep track of current best classifier at each iteration and exit after a maximum number of iterations
        while err!=0 and iterations<self.iterations:    
            # use z to store Xa.w value
            z = np.dot(Xa, w)

            mask_p = (KC>0)&(z<=0)
            mask_n = (KC<0)&(z>=0)

            err = np.sum(mask_p)+np.sum(mask_n)

            if err < best_err:
                best_err = err
                best_w = w.copy()
                
            # update weight
            for i in np.arange(k):
                w[:, i] = w[:,i] + np.sum(mask_p[:,i].reshape(-1,1)*Xa, axis=0)
                w[:, i] = w[:,i] - np.sum(mask_n[:,i].reshape(-1,1)*Xa, axis=0)
            
            iterations += 1

        self.w = best_w

    def predict(self, query):
        labels = self.Keslerlabel
        query_a = self._Xa_construction(query)

        z = np.dot(query_a, self.w)

        y = np.argmax(z, axis=1)

        pred = np.zeros_like(y)

        for k, v in labels.items():
            pred[y==k]=v

        return pred

"""
# Usage
model = perceptron(30000)
model.fit(X, y)
pred = model.predict(X)
"""
