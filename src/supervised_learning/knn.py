from supervised_learning.utils.operation import minkowski_distance
import numpy as np 

class KNN():
    '''
    k: int 
        assume the default hyperparameter k = 5 
    p: int 
        the chosen minkowski distance p parameter 
            p = 1: manhattan distance 
            p = 2: euclidean distance 
            p = inf: chebysev distance 
   ''' 
    
    def __init__(self, k: int, p: int): 
        self.k = k 
        self.p = p 
    

    def _vote(self, neighbors): 
        ''' 
        most common class among its neighbors 
        '''
        counts = np.bincount(neighbors.astype('int'))
        return counts.argmax() 
    
    def predict(self, X_test, X_train, y_train): 
        '''
        Returns a numpy array that is the size of X_test  
        '''

        y_pred = np.empty(X_test.shape[0])

        for i, sample in enumerate(X_test): 
            idx = np.argsort([minkowski_distance(sample, x, self.p) for x in X_train])[:self.k]
            # we now have the neareste neighbors 
            knn = np.array([y_train[i] for i in idx]) 
            y_pred[i] = self._vote(knn) 

        return y_pred 
