import math

import numpy as np

from ml_lib.model.base import BaseModel
from ml_lib.utils.functions import Sigmoid


class Regression(BaseModel): 
    def __init__(self, learning_rate=.1, gradient_descent=True): 
        self.param = None 
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid() 

    def _init_params(self, X): 
        # create feature matrix 
        n_features = np.shape(X)[1]  # 1 is column 
        # limit the feature squeezed by the sigmoid function to sqrt(n) 
        limit = 1 / math.sqrt(n_features) 

        # create the param matrix 
        self.param = np.random.uniform(-limit, limit, (n_features,)) 

    def fit(self, X, y): 
        self._init_params(X) 
        for i in range(self.get_params().get("n_iteration")):               
            if (self.gradient_descent): 
                return 
        return 




