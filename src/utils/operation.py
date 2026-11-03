from __future__ import division 
import numpy as np 

def minkowski_distance(A, B, p):
    ''' 
    p is the degree of distance used, i.e. 
    p = 1:  manhattan distance
    p = 2:euclidean distance 
    p -> inf: chebyshev distance  
    Returns the minkowski distance of x1 and x2 
    '''
    return np.power(np.sum(np.abs(np.array(A) - np.array(B)) ** p), 1/p)


