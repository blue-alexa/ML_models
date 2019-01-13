import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = None

class DecisionTree:
    def __init__(self, min_samples_split, criterion):
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def _variance_impurity(self, y):
        """
        Calculate impurity for binary class.
        Variance impurity: i(N) = P(ω1)P(ω2)
        P is prevalence of that class. 
        Input:
        -y: class labels of datasets, with shape (N,)
        -class_values: class value for positive and negative class, list [-1, 1]
        
        Output: 
        -impurity: impurity measurement of y, float.
        """
        if len(np.unique(y))==1:
            return 0
        else:
            class1, class2 = np.unique(y)
            c1 = np.sum(y==class1)
            c2 = np.sum(y==class2)
            impurity = c1*c2/((c1+c2)**2)
            return impurity
    
    def _gini_impurity(self, y):
        """
        Calculate impurity for multi classes. Use Gini method. Impurity = 1-
        Gini impurity, 1 − sum(P(ωj)**2)
        Input:
        -y: class labels of datasets, with shape (N,)
        
        Output: 
        -impurity: impurity measurement of y, float.
        """
        class_values = np.unique(y)
        impurity = 1
        total_samples = len(y)

        for value in class_values:
            impurity -= (np.sum(y==value)/total_samples)**2
        return impurity

    def impurity_measure(self, y):
        """choose proper criterion for impurity measurement.
        type: variance, gini

        Input:
        -y: class labels of datasets, with shape (N,)
        -self.criterion: "variance" or "gini" 

        Output:
        -impurity: impurity measurement of y, float.
        """

        if self.criterion == "variance":
            impurity = self._variance_impurity(y)
        elif self.criterion == "gini":
            impurity = self._gini_impurity(y)
        else:
            raise ValueError('criterion can only be "vairance" or "gini".')

        return impurity

    def single_feature_split(self, X, y):
        """
        find the best split location to split a single feature, bi-class or multi-class
        
        Input:
        -X: one feature data, with shape of (N,)
        -y: class label, with shape of (N,)
        -min_samples_split: The minimum number of samples required to split an internal node   
        
        Output:
        -best_purity: best purity achieved by optimal split
        -best_value: the value for optimal split (left<value; right>=value)
        -subsets: left and right subsets after split
        """
        N_samples = len(y)

        # calculate purity before any splitting
        start_impurity = self.impurity_measure(y)
        
        # initialize best_purity, best_value
        best_impurity = start_impurity
        best_value = np.nan
        
        # if X is too short for splitting, or already is pure, or feature is singular value, no split should be done.
        if len(X)<self.min_samples_split or len(np.unique(y))==1 or len(np.unique(X))==1:
            return (best_impurity, best_value, 0)
        
        # find out all potential splitting locations
        split_values = np.sort(np.unique(X), axis=None)[1:]
        
        # sort X and y according to value of X (from small to large)
        ind = np.argsort(X)
        Xsorted = X[ind]
        ysorted = y[ind]  
        
        # try every possible split value
        for value in split_values:
            # purity of left subset
            l_y = ysorted[Xsorted<value]
            l_impurity = self.impurity_measure(l_y)
            
            # purity of right subset
            r_y = ysorted[Xsorted>=value]
            r_impurity = self.impurity_measure(r_y)
            
            impurity = (len(l_y)/N_samples)*l_impurity + (len(r_y)/N_samples)*r_impurity
            
            if impurity<best_impurity:
                best_impurity = impurity
                best_value = value
        
        impurity_reduction = start_impurity - best_impurity
        
        return (best_impurity, best_value, impurity_reduction)        

    def dataset_split(self, X, y):
        """
        find the best feature and best split location in datasets, bi-class or multi-class
        
        Input:
        -X: training data, with shape of (N,d)
        -y: class label, with shape of (N,)
        
        Output:
        -best_feature: best feature to split
        -best_impurity: best impurity achieved by optimal split
        -best_value: the value for optimal split (left<value; right>=value)
        -subsets: left and right data subsets after split
        """
        n_samples, n_features = X.shape
        impurities = np.ones((n_features,)).astype('float64')
        values = np.zeros((n_features,)).astype('float64')
        values.fill(np.nan)
        impurity_reductions = np.ones((n_features,)).astype('float64')
        impurity_reductions.fill(np.nan)
        
        # calculate best_purity for all features
        for i in range(n_features):
            impurities[i], values[i], impurity_reductions[i] = self.single_feature_split(X[:, i], y)
        
        best_feature = np.argmin(impurities)
        best_impurity = impurities[best_feature]
        best_value = values[best_feature]
        best_impurity_reduction = impurity_reductions[best_feature]
        
        if np.isnan(best_value):
            subsets = {}
        else:
            subsets = self.split_to_subsets(X, y, best_feature, best_value)
        
        return best_feature, best_impurity, best_value, best_impurity_reduction, subsets
    
    def split_to_subsets(self, X, y, feature_index, value):
        """
        Split X, y according to specific value in specific feature. 
        In the left subset, all data of specific feature have value < specific value
        In the right subset, all data of specific feature have value >= specific value
        
        Input:
        -X: training data, with shape of (N,d)
        -y: class label, with shape of (N,)
        -feature_index: the index of feature to be split, in range of int [0, d)
        -value: the value of split boundary. 
        """
        lmask = np.where(X[:, feature_index]<value)
        yleft = y[lmask]
        Xleft = X[lmask]
        
        rmask = np.where(X[:, feature_index]>=value)
        yright = y[rmask]
        Xright = X[rmask]
        
        return {'left': (Xleft, yleft), 'right': (Xright, yright)}
    
    def leaf_pred(self, y):
        """
        find the class id of a leaf node.
        Input:
        -y: class labels, of shape (N,)
        
        Output:
        -class_id: predicted class for that leaf node. int
        """    
        return (Counter(y).most_common(1)[0][0])  

    def grow_tree(self, X, y, root):
        
        # calculate best split
        best_feature, best_purity, best_value, best_impurity_reduction, subsets = self.dataset_split(X, y)
        
        if not subsets:
            class_id = self.leaf_pred(y)
            root.value = {'type':'leaf', 'pred': class_id, 'impurity_reduction': best_impurity_reduction}
        else:
            root.value = {'type':'rule', 'rule':(best_feature, best_value), 'impurity_reduction': best_impurity_reduction}
            root.left = TreeNode()
            self.grow_tree(subsets['left'][0], subsets['left'][1], root.left)
            root.right = TreeNode()
            self.grow_tree(subsets['right'][0], subsets['right'][1], root.right)

    def fit(self, X, y):
        root = TreeNode()
        self.grow_tree(X, y, root)
        self.root = root

    def print_tree(self):
        """
        print tree nodes level by level.
        
        Input:
        -root: decision tree root node.
        """
        if self.root:    
            queue = list()
            queue.append(self.root)
            
            while queue:
                node = queue[0]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if node.value['type'] == 'leaf':
                    print (node.value['type'], node.value['pred'], node.value['impurity_reduction'])
                if node.value['type'] == 'rule':
                    print (node.value['type'], node.value['rule'], node.value['impurity_reduction'])
                queue = queue[1:] 

    def predict_single_data(self, X):
        """
        predict class label for a single data.
        Input:
        -X: data, with shape(d, )
        
        Output:
        -label
        """
        label = None
        node = self.root
        
        while True:
            if node.value['type'] == 'leaf':
                return node.value['pred']
            else:
                feature, value = node.value['rule']
                if X[feature]<value:
                    node = node.left
                else:
                    node = node.right

    def predict(self, query):
        """    
        predict class label for query.
        Input:
        -query: query data, with shape of (N,d)
        
        Output:
        -pred: predicted class for query, with shape (N,)
        """
        pred = np.zeros((len(query),))
        pred.fill(np.nan)

        for i in np.arange(len(pred)):
            pred[i] = self.predict_single_data(query[i])
        
        return pred

# usage binary class
# fit data
tree = DecisionTree(2, "variance")
tree.fit(X, y)
tree.print_tree()

# check accuracy of training data
pred = tree.predict(X)
# evaluate predictions
TP = np.sum((y==1)&(pred==1))
TN = np.sum((y==-1)&(pred==-1))
FP = np.sum((y==-1)&(pred==1))
FN = np.sum((y==1)&(pred==-1))
print (TP, TN, FP, FN)

# Decision boundary
query = np.sort(np.unique(X), axis=None).astype('int64')
query = query.reshape(-1,1)
pred_q = tree.predict(query)
print (np.hstack((query, pred_q.reshape(-1,1))))

# Usage 6 class classification
# fit training data
tree = DecisionTree(66, "gini")
tree.fit(X, y)
tree.print_tree()

# check accuracy of training data
pred = tree.predict(X)

# confusion matrix of 6 class
conf_mat_6 = np.zeros((6,6))

for i in np.arange(6):
    for j in np.arange(6):
        conf_mat_6[i][j]=np.sum((y==i)&(pred==j))
        
print (conf_mat_6.astype('int64'))