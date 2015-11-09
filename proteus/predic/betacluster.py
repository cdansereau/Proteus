__author__ = 'Christian Dansereau'

from proteus import matrix
from ..matrix import tseries as ts
import pandas as pd
import clustering as cls
from sklearn import linear_model
import numpy as np
import prediction
from sklearn import cross_validation
from proteus.predic import stability 

class BetaCluster:
    '''
    Prediction tool for multiscale functional neuro imaging
    Beta clustering
    '''

    def __init__(self, x, y, n_cluster, k_feature=0, samp_ratio=0.5, nsample=100):
        return self.fit(x, y, n_cluster, k_feature, samp_ratio, nsample)
    
    def fit(self, x, y, n_cluster, k_feature=0, samp_ratio=0.5, nsample=100):
        '''
        x: (samples,features)
        y: (samples,covar) first colum is the covar of interest the rest are covar to include in the model but are not of interest
        k_features: k best feature based on a ttest on each features, the k best features are retained. A stability analysis is also performed on the selected features
        samp_ratio: (default 0.5) subsample for the stability analysis
        nsample: (default 100) number of iteration for the stability analysis 
        '''
        self.k_feature = k_feature
        self.samp_ratio = samp_ratio
        self.nsample = nsample
 
        # GLM of the groups
        clf = linear_model.LinearRegression()
        clf.fit (x, y)
        # get the beta vector of the target
        beta = ts.vec2mat(clf.coef_,0) # Beta matrix
        
        # Hierachical clustering on the beta matrix
        self.ind = cls.hclustering(beta, n_cluster)
        
        # Stability estimation
        if k_feature != 0:
            self.hr_mat = stability.itStability(x,y,self.ind,k=k_feature,samp_ratio=samp_ratio,nsample=nsample)
            # Do a feature selection
            bc_x = self.bc_transform(x)
            self.selectidx = stability.getkBest(bc_x,y,k_feature)
            
        
        # Training and prediction
        #skf = cross_validation.StratifiedKFold(y, n_folds=10)
        #score = prediction.basicconn(skf, x_subf, y)
        #print score
        #prediction.multisplit(skf, vec_features.values, y)

    def bc_transform(self,x):
        # average resulting new partition
        # Obtain the new individual conectomes in a vector format
        x_subf = []
        for i1 in range(0, x.shape[0]):
            m = ts.vec2mat(x[i1, :])
            m = cls.part(m, self.ind)
            if len(x_subf)==0:
                x_subf = ts.mat2vec(m,include_diag=True)
            else:
                x_subf = np.vstack((x_subf,ts.mat2vec(m,include_diag=True))) 
        return x_subf
    
    def transform(self,x):
        bc_x = self.bc_transform(x)
        # check if we do a feature selection
        if self.k_feature != 0:
            return x[:,self.selectidx]
        return bc_x

