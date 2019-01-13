import numpy as np

class Ensemble_linear_classifier:
    """
    Implementation of ensemble linear classifiers algorithm. None linear transformation using Tanh.

    Input:
    -X: training data, with shape (N, d)
    -query: query data, with shape (N, d)

    """
    def __init__(self, n_estimators=30, random_state=42):
        """
        n_estimators: number of linear classifier
        random_state: for generating random seeding
        Keslerlabel: dictionary to show corresponding class value in Kesler construction
        w: weight
        e: error weight for ensembled classifier
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.Keslerlabel = None
        self.w = None
        self.e = None

    def _Kesler_construction(self, X, y):
        # convert y to Kesler constructions, positive class +1, negative class -1
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
        # construct weight
        M = self.n_estimators
        N, d = X.shape

        # generate random weights and normalize
        np.random.seed(self.random_state)
        w = np.random.uniform(-1,1,(M,d))
        w = w/np.linalg.norm(w, axis=1, keepdims=True)

        # random select M samples from X
        np.random.seed(self.random_state)
        ind = np.random.choice(len(X), size=M)
        Xk=X[ind]
        w0 = -np.sum(Xk*w, axis=1, keepdims=True)
        w = np.hstack((w0, w))
        self.w = w

        Xa = self._Xa_construction(X)
        KC = self._Kesler_construction(X, y)

        # transform X from d-dim feature to M-dim feature
        C = np.tanh(np.dot(Xa, w.T))
        Ca = self._Xa_construction(C)

        # calculate error weights for ensembled classifier
        e = np.dot(np.linalg.pinv(Ca), KC)
        self.e = e

    def predict(self, query):
        labels = self.Keslerlabel
        query_a = self._Xa_construction(query)

        C = np.tanh(np.dot(query_a, self.w.T))
        Ca = self._Xa_construction(C)

        y = np.argmax(np.dot(Ca, self.e), axis=1)

        pred = np.zeros_like(y)

        for k, v in labels.items():
            pred[y==k]=v

        return pred

"""
# Usage
clf = Ensemble_linear_classifier(n_estimators=60, random_state=42)
clf.fit(X, y)
pred = clf.predict(query)
"""