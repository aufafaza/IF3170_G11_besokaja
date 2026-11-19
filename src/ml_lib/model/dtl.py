from __future__ import annotations

from collections import Counter 
import numpy as np 
from sklearn.base import ClassifierMixin

from ml_lib.model.base import BaseModel

class TreeNode(): 
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None): 
        self.feature=feature 
        self.threshold = threshold 
        self.left = left 
        self.right = right 
        self.value = value 

    def is_leaf_node(self): 
        return self.value is not None 


class DecisionTree(ClassifierMixin, BaseModel):
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None): 
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        self.n_features = n_features 
        self._n_features = None
        self.root=None 

    def fit(self, X, y) -> DecisionTree:
        X = np.asarray(X)
        y = np.asarray(y)
        self._n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        
        self.root = self._grow_tree(X, y) 
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def _grow_tree(self, X, y, depth=0): 
        n_samples, n_feats = X.shape 
        n_labels = len(np.unique(y)) 

        # check stopping criteria 
        if (depth >= self.max_depth or n_labels==1 or n_samples < self.min_samples_split): 
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        # find the best split (gain) 
        feat_indices = np.random.choice(n_feats, self._n_features, replace=False)

        best_feature, best_thresh_ = self._best_split(X, y, feat_indices) 
        if best_feature is None:
            return TreeNode(value=self._most_common_label(y))

        # create child nodes 
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh_)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return TreeNode(best_feature, best_thresh_, left, right)
   
    def _best_split(self, X, y, feat_indices): 
        best_gain = -1 

        split_idx, split_threshold = None, None  

        for feat_i in feat_indices: 
            X_col = X[:, feat_i] 
            thresholds = np.unique(X_col)[:-1]

            for thr in thresholds: 
                # information gain 
                gain = self._calculate_gain(y, X_col, thr) 
                # select the highest gain 
                if (gain > best_gain): 
                    best_gain = gain 
                    split_idx = feat_i 
                    split_threshold = thr 

        return split_idx, split_threshold 
    
    def _calculate_gain(self, y, col, thr): 
        # entropy(parent) - [weighted_avg].E(children)
        
        # parent entropy 
        parent_entropy = self._entropy(y) 

        # create children 
        left_idxs, right_idxs = self._split(col, thr)
        if len(left_idxs) == 0 or len(right_idxs) == 0: 
            return 0 

        # calculate weighted entropy children 
        n = len(y) 
        n_l, n_r = len(left_idxs), len(right_idxs) 
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r 

        # calculate information gain 

        return parent_entropy - child_entropy 
    
    def _split(self, col, split_thr): 
        left_idxs = np.argwhere(col<=split_thr).flatten() 
        right_idxs = np.argwhere(col>split_thr).flatten()  
        
        return left_idxs, right_idxs 
    def _entropy(self, y): 
        # -sum(p(X).base2log(p(X)) 
        _, hist = np.unique(y, return_counts=True)
        p_s = hist / len(y) 
        
        return -np.sum([p * np.log(p) for p in p_s if p > 0])


    def _most_common_label(self, y): 
        # calculate gain 
        counter = Counter(y) 
        value = counter.most_common(1)[0][0] 
        return value

    def predict(self, X) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("DecisionTree must be fitted before calling predict")

        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.root) for x in X]) 

    def _traverse_tree(self, x, node): 
        if node.is_leaf_node(): 
            return node.value
        
        if x[node.feature] <= node.threshold: 
            return self._traverse_tree(x, node.left) 

        return self._traverse_tree(x, node.right) 


    
