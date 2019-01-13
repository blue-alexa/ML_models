"""
Dual Perceptron
1. Start with zero counts (alpha)
2. Pick up training instances one by one
3. Try to classify xn,
y = argmax(Sigma ai,y K(xi,xn)) bias?
4. If correct, no change!
5. If wrong: lower count of wrong class (for this instance), raise count of right class (for this instance)
ay,n = ay,n-1
ay*,n = ay*,n+1
"""

def Linear_kernel(X, Xp):
    """
    K(Xi, Xj)=k[i,j]; K(X, Xi)=k[:,i]
    
    Input:
    -X, with shape (N1,d)
    -Xp, with shape (N2,d)

    Output:
    -k, with shape (N1, N2)
    """
    k = np.dot(X, Xp.T)
    return k

def Quadratic_kernel(X, Xp):
    """
    K(Xi, Xj)=k[i,j]; K(X, Xi)=k[:,i]
    
    Input:
    -X, with shape (N1,d)
    -Xp, with shape (N2,d)

    Output:
    -k, with shape (N1, N2)
    """
    k = (np.dot(X, Xp.T)+1)**2
    return k

def RBF_kernel(X, Xp, tau):
    """
    K(Xi, Xj)=k[i,j]; K(X, Xi)=k[:,i]
    
    Input:
    -X, with shape (N1,d)
    -Xp, with shape (N2,d)

    Output:
    -k, with shape (N1, N2)
    """
    k1 = np.sum(X**2, axis=1, keepdims=True)
    k2 = np.sum(Xp**2, axis=1, keepdims=True)
    k12 = np.dot(X, Xp.T)
    k = np.exp(-(k1+k2.T-2*k12)/(2*tau**2))
    return k

def Kesler_construction(X, y):
    # convert y to Kesler constructions
    class_values = np.unique(y)
    class_num = len(class_values)
    yc = np.zeros_like(y)

    for i, value in enumerate(class_values):
        yc[y==value]=i

    KC = np.zeros((len(X), class_num))
    KC.fill(-1)
    KC[range(len(X)), yc]=1

    #Keslerlabel = {ind:value for ind, value in enumerate(class_values)}

    return KC

def Xa_construction(X):
    # add x0=1
    Xa = np.hstack((np.ones((len(X),1)), X))
    return Xa

def fit(X, y):
    KC = Kesler_construction(X, y)
    y_KC = np.argmax(KC, axis=1)

    # N is number of training samples, d is number of features. 
    N, d = X.shape
    # k is number of classes
    _, k = KC.shape
    
    # initialize alphas, columns are alpha for one class
    alpha = np.zeros((N, k))
    
    # Use quadratic_kernel
    Ker = Quadratic_kernel(X, X)

    iterations = 0
    err = N
    best_err = err
    best_alpha = alpha

    while err!=0 and iterations<10000: 
        # update all training sample using matrix formate
        pred = np.argmax(np.dot(Ker.T, alpha), axis=1)
        row_change = np.where(pred != y_KC)
        col_change = y_KC[row_change]
        alpha[row_change, :] -= 1
        alpha[row_change, col_change] += 2

        # check error rate:
        pred = np.argmax(np.dot(Ker.T, alpha), axis=1)
        err = np.sum(pred!=y_KC)
        
        if err < best_err:
            best_err = err
            best_alpha = alpha.copy()

        iterations+=1
    """
    # update one training sample Xi
    pred = np.argmax(np.dot(Ker[:, i, np.newaxis].T, alpha))
    if pred != y_KC[i]:
        alpha[i, :] -= 1
        alpha[i, y[i]] += 2
    """




