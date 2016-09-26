__author__ = ' iChristian Dansereau'

import numpy as np
from sklearn.cluster import KMeans
from proteus.predic import clustering as cls
from proteus.matrix import tseries as ts
from proteus.predic import prediction
from proteus.predic import subtypes
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MeanShift
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut, LeavePOut, StratifiedKFold
from sklearn.metrics import classification_report,accuracy_score

from nistats import glm as nsglm
import statsmodels.stats.multitest as smm
import multiprocessing
import time

class StackedPrediction:
    '''
    2 Level prediction
    '''

    def makeNewModel(self,x,y):
        clf = LogisticRegression(C=1.,class_weight='balanced',penalty='l2')
        #clf = LinearSVC(C=1.,class_weight='balanced')
        param_grid = dict(C=(10**np.arange(0.,-1.5,-0.5)))
        gridclf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(y,n_folds=4), n_jobs=-1,scoring='accuracy')
        gridclf.fit(x,y)
        return gridclf.best_estimator_

    def fitMulti(self,x,y):
        self.clf_list = []
        for ii in range(x.shape[1]):
            self.clf_list.append(self.makeNewModel(x[:,ii,:], y))


    def predictMulti(self,x):
        pred = []
        for ii in range(len(self.clf_list)):
            if len(x.shape) == 2:
                pred.append(self.clf_list[ii].predict(x[ii,:].reshape(1,-1)))
            else:
                pred.append(self.clf_list[ii].predict(x[:,ii,:]))
       #     print self.clf_list[ii]
       #     print (self.clf_list[ii].coef_**2.).sum()
        pred = np.array(pred)
        if len(pred.shape) == 2:
            pred = np.swapaxes(pred,0,1)
        else:
            pred = np.swapaxes(pred,1,2)[:,:,0]
        return pred

    def fit(self,x,y):
        # Stage 1
        hm = self.estimate_hitmiss(x, y)
        #self.fitMulti(x,y)
        #hm = self.predictMulti(x)

        print (hm == np.tile(y,(hm.shape[1],1)).T).mean(axis=0)
        # Stage 2
        clf2 = LogisticRegression(C=1.,class_weight='balanced',penalty='l2',max_iter=3)
        param_grid = dict(C=(10**np.arange(0.,-.5,-0.25)))

        gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(y,n_folds=4), n_jobs=-1,scoring='accuracy')
        gridclf.fit(hm,y)

        self.clf_stack = gridclf.best_estimator_
        print 'stack',self.clf_stack
        print 'stack',(self.clf_stack.coef_**2.).sum()

    def cv(self,X,y,gs=4):
        k=1
        skf = StratifiedKFold(y, n_folds=gs)
        scores = []
        for train_index, test_index in skf:
            print('Fold: '+str(k)+'/'+str(gs))
            k+=1
            # train
            self.fit(X[train_index], y[train_index])
            # test
            y_pred = self.predict(X[test_index])
            scores.append((y_pred==y[test_index]).sum()/(1.*y[test_index].shape[0]))
            print classification_report(y[test_index], y_pred)

        scores = np.array(scores)
        return [scores,scores.mean(),scores.std()]

    def predict(self,x):
        l1_results = self.predictMulti(x)
        return self.clf_stack.predict(l1_results)


        #y_pred1 = self.clf1.predict(xw)
        #y_pred2 = self.clf2.decision_function(xw)
        #return np.array([y_pred1,y_pred2]).T

    def estimate_hitmiss(self,x,y):
        # Perform a LOO to estimate the actual HM
        label=1
        hm_results = []
        predictions =[]
        for i in range(len(y)):
            train_idx = np.array(np.hstack((np.arange(0,i),np.arange(i+1,len(y)))),dtype=int)
            self.fitMulti(x[train_idx,:,:],y[train_idx])
            hm_results.append((self.predictMulti(x[i,:,:]) == y[i]).astype(int))
            #predictions.append(clf.predict(x[i,:,:].reshape(1,-1)))

        #predictions = np.array(predictions)
        hm_results = np.array(hm_results)
        hm_results = np.swapaxes(hm_results,1,2)[:,:,0]

        #print hm_results.shape
        self.fitMulti(x,y)
        return hm_results#, predictions[:,0]

