__author__ = 'Christian Dansereau'

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
from sklearn.svm import SVC,LinearSVC,l1_min_c
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut, LeavePOut, StratifiedKFold,StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from nistats import glm as nsglm
import statsmodels.stats.multitest as smm
import multiprocessing
import time
from proteus.io import sbp_util



def compute_loo_parall((net_data_low_main,y,confounds,n_subtypes,train_index,test_index)):
    my_sbp = sbp()
    my_sbp.fit(net_data_low_main[train_index,...],y[train_index],confounds[train_index,...],n_subtypes,verbose=False)
    tmp_scores = my_sbp.predict(net_data_low_main[test_index,...],confounds[test_index,...])
    return np.hstack((y[test_index],tmp_scores[0][0],tmp_scores[1][0]))

class SBP:
    '''
    Pipeline for subtype base prediction
    '''
    def __init__(self,verbose=True,dynamic=True,gamma=0.999,stage1_model_type='svm',nSubtypes=7,mask_part=[]):
        self.verbose=verbose
        self.dynamic = dynamic
        self.gamma = gamma
        self.mask_part = mask_part
        self.stage1_model_type = stage1_model_type
        self.nSubtypes = nSubtypes

    def get_w(self,x,confounds):
        ### extract w values
        W = []
        for ii in range(len(self.st_crm)):
            W.append(self.st_crm[ii][1].compute_weights(self.st_crm[ii][0].transform(confounds,x[:,ii,:]),mask_part=self.mask_part))
        xw = np.hstack(W)
        return subtypes.reshapeW(xw)

    def get_w_files(self,files_path,subjects_id_list,confounds):
        ### extract w values
        W = []
        for ii in range(len(self.st_crm)):
            x_ref = sbp_util.grab_rmap(subjects_id_list,files_path,ii,dynamic=False)
            ## compute w values
            W.append(self.st_crm[ii][1].compute_weights(self.st_crm[ii][0].transform(confounds,x_ref),mask_part=self.mask_part))
            del x_ref

        xw = np.hstack(W)
        return subtypes.reshapeW(xw)


    def fit(self,x_dyn,confounds_dyn,x,confounds,y,extra_var=[]):

        if self.verbose: start = time.time()
        ### train subtypes
        self.st_crm = []
        for ii in range(x.shape[1]):
            crm = prediction.ConfoundsRm(confounds_dyn,x_dyn[:,ii,:])
            # st
            st=subtypes.clusteringST()
            st.fit_network(crm.transform(confounds_dyn,x_dyn[:,ii,:]),nSubtypes=self.nSubtypes)
            self.st_crm_s2.append([crm,st])

        ### extract w values
        xw = self.get_w(x,confounds)
        print 'xw sub data',xw[0,:]
        if self.verbose: print("Subtype extraction, Time elapsed: {}s)".format(int(time.time() - start)))

        ### Include extra covariates
        if len(extra_var)!=0:
            all_var = np.hstack((xw,extra_var))
        else:
            all_var = xw

        ### prediction model
        if self.verbose: start = time.time()
        tlp = TwoLevelsPrediction(self.verbose,stage1_model_type=self.stage1_model_type,gamma=self.gamma)
        tlp.fit(all_var,all_var,y)
        if self.verbose: print("Two Levels prediction, Time elapsed: {}s)".format(int(time.time() - start)))

        ### save parameters
        self.tlp = tlp

    def fit_files_st(self,files_path_st,subjects_id_list_st,confounds_st,files_path,subjects_id_list,confounds,y,n_seeds,extra_var=[]):
        '''
        Use a list of subject IDs and search for them in the path, grab the results per network.
        Same as fit_files() except that you can train and test on different set of data
        '''
        if self.verbose: start = time.time()
        ### train subtypes
        self.st_crm = []
        #for ii in [5,13]:#range(x.shape[1]):
        xw = []
        for ii in range(n_seeds):
            print('Train seed '+str(ii+1))
            if self.dynamic:
                [x_dyn,x_ref] = sbp_util.grab_rmap(subjects_id_list_st,files_path_st,ii,dynamic=self.dynamic)
                confounds_dyn = []
                for jj in range(len(x_dyn)):
                    confounds_dyn.append((confounds_st[jj],)*x_dyn[jj].shape[0])
                confounds_dyn = np.vstack(confounds_dyn)
                x_dyn = np.vstack(x_dyn)
            else:
                x_ref = sbp_util.grab_rmap(subjects_id_list_st,files_path_st,ii,dynamic=self.dynamic)
                x_dyn = x_ref
                confounds_dyn = confounds_st

            del x_ref
            ## regress confounds
            crm = prediction.ConfoundsRm(confounds_dyn,x_dyn)
            ## extract subtypes
            st=subtypes.clusteringST()
            st.fit_network(crm.transform(confounds_dyn,x_dyn),nSubtypes=self.nSubtypes)
            self.st_crm.append([crm,st])
            del x_dyn

        # compute the W
        xw = self.get_w_files(files_path,subjects_id_list,confounds)
        if self.verbose: print("Subtype extraction, Time elapsed: {}s)".format(int(time.time() - start)))

        ### Include extra covariates
        if len(extra_var)!=0:
            all_var = np.hstack((xw,extra_var))
        else:
            all_var = xw

        ### prediction model
        if self.verbose: start = time.time()
        self.tlp = TwoLevelsPrediction(self.verbose,stage1_model_type=self.stage1_model_type,gamma=self.gamma)
        self.tlp.fit(all_var,all_var,y)
        if self.verbose: print("Two Levels prediction, Time elapsed: {}s)".format(int(time.time() - start)))


    def fit_files(self,files_path,subjects_id_list,confounds,y,n_seeds,extra_var=[]):
        '''
        use a list of subject IDs and search for them in the path, grab the results per network
        '''
        self.fit_files_st(files_path,subjects_id_list,confounds,files_path,subjects_id_list,confounds,y,n_seeds,extra_var)

    def predict_files(self,files_path,subjects_id_list,confounds,extra_var=[]):
        xw = self.get_w_files(files_path,subjects_id_list,confounds)
        return self.predict(xw,[],extra_var,skip_confounds=True)

    def predict(self,x,confounds,extra_var=[],skip_confounds=False):

        if skip_confounds:
            xw = x
        else:
            xw = self.get_w(x,confounds)

        ### Include extra covariates
        if len(extra_var)!=0:
            all_var = np.hstack((xw,extra_var))
        else:
            all_var = xw

        ### prediction model
        return self.tlp.predict(all_var,all_var)

    def score_files(self,files_path,subjects_id_list,confounds,y,extra_var=[]):
        res = self.predict_files(files_path,subjects_id_list,confounds,extra_var)
        l1_y_pred = (res[:,0]>0).astype(int)
        risk_mask = res[:,1]>0
        right_cases = accuracy_score(y[risk_mask],l1_y_pred[risk_mask])
        left_cases = accuracy_score(y[~risk_mask],l1_y_pred[~risk_mask])
        self.res    = np.vstack((y,res[:,0],res[:,1]))
        self.scores = (accuracy_score(y,l1_y_pred),left_cases,right_cases)
        return self.scores

    def score(self,x,confounds,y,extra_var=[]):
        res = self.predict(x,confounds,extra_var)
        l1_y_pred = res[:,0]
        risk_mask = res[:,1]>0
        right_cases = accuracy_score(y[risk_mask],res[risk_mask,0])
        left_cases = accuracy_score(y[~risk_mask],res[~risk_mask,0])
        self.res    = np.vstack((y,res[:,0],res[:,1]))
        self.scores = (accuracy_score(y,l1_y_pred),left_cases,right_cases)
        return self.scores

    def estimate_acc(self,net_data_low_main,y,confounds,n_subtypes,verbose=False):

        sss = LeaveOneOut(len(y))
        # scores: y, y_pred, decision_function
        self.scores = []
        k=0
        for train_index, test_index in sss:
            k+=1
            print('Fold: '+str(k)+'/'+str(len(y)))
            self.fit(net_data_low_main[train_index,...],y[train_index],confounds[train_index,...],n_subtypes=n_subtypes,verbose=False,flag_feature_select=False)
            tmp_scores = self.predict(net_data_low_main[test_index,...],confounds[test_index,...])
            self.scores.append(np.hstack((y[test_index],tmp_scores[0][0],tmp_scores[0][1])))
        self.scores = np.array(self.scores)

    def estimate_acc_multicore(self,net_data_low_main,y,confounds,n_subtypes,verbose=False):
        taskList_loo = []
        sss = LeaveOneOut(len(y))
        # scores: y, y_pred, decision_function
        self.scores = []
        k=0
        for train_index, test_index in sss:

            taskList_loo.append((net_data_low_main,y,confounds,n_subtypes,train_index,test_index))

        pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 2)) #Don't use all my processing power.
        r2 = pool.map_async(compute_loo_parall, taskList_loo, callback=self.scores.append)  #Using fxn "calculate", feed taskList, and values stored in "results" list
        r2.wait()
        pool.terminate()
        pool.join()
        self.scores = np.array(self.scores)

class TwoLevelsPrediction:
    '''
    2 Level prediction
    '''
    def __init__(self,verbose=True,stage1_model_type='svm',gamma=0.999):
        self.verbose=verbose
        self.stage1_model_type = stage1_model_type
        self.gamma = gamma


    def auto_gamma(self,proba,gamma,thresh=0.15):
        while (np.mean(proba>gamma)<=thresh):
            gamma -= 0.01

        return (proba>gamma).astype(int),gamma

    def fit(self,xw,xwl2,y,gs=4,retrain_l1=False):
        print 'Stage 1'
        if self.stage1_model_type == 'logit':
            clf = LogisticRegression(C=1,class_weight='balanced',penalty='l2',max_iter=300)
        elif self.stage1_model_type == 'svm':
            #clf = SVC(kernel='linear', class_weight='balanced', C=.1,probability=False)
            clf = SVC(C=1.,cache_size=500,kernel='linear',class_weight='balanced',probability=False)
        elif self.stage1_model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=20,class_weight='balanced')

        # Stage 1
        #param_grid = dict(C=(np.array([5,3,1])))
        if self.stage1_model_type == 'logit':
            #param_grid = dict(C=(10**np.arange(1.,-2.,-0.5)))
            #param_grid = dict(C=(np.logspace(-.2, 1., 15)))
            #param_grid = dict(C=(np.arange(3,1,-0.5)))
            param_grid = dict(C=(5,5.0001))
        elif self.stage1_model_type =='svm':
            param_grid = dict(C=(np.arange(3.5,0.,-0.5)))
            param_grid = dict(C=(1.,1.00001))
            #param_grid = dict(C=(np.logspace(-1.5, 0, 10)))
            #param_grid = dict(C=(np.arange(2.,0.5,-0.05)))
            #param_grid = dict(C=(np.array([0.01, 0.1, 1, 10, 100, 1000])))
        elif self.stage1_model_type == 'rf':
            param_grid = dict(n_estimators=(20,10))

        gridclf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(n_splits=gs), n_jobs=-1,scoring='accuracy')
        gridclf.fit(xw,y)
        self.clf1 = gridclf.best_estimator_
        if self.verbose:
            print self.clf1
            #print self.clf1.coef_
        #hm_y,y_pred_train = self.estimate_hitmiss(xw,y)
        hm_y,proba = self.suffle_hm(xw,y,gamma=self.gamma,n_iter=100)
        hm_y,auto_gamma = self.auto_gamma(proba,self.gamma)
        self.auto_gamma = auto_gamma
        if self.verbose: proba
        if self.verbose: print 'Average hm score', np.mean(hm_y)
        #self.clf3 = SVC(C=1.,cache_size=500,kernel='linear',class_weight='balanced',probability=False)
        #gamma=0.5
        #print 'n stage3 ',(proba>gamma).sum()
        #self.clf3.fit(xw[proba>gamma,:],y[proba>gamma])
        #if retrain_l1:
        #    self.clf1 = self.clf3
        print 'Stage 2'
        #Stage 2
        min_c = l1_min_c(xwl2,hm_y,loss='log')
        #clf2 = LogisticRegression(C=10**0.1,class_weight=None,penalty='l2',solver='sag')
        #clf2 = LogisticRegression(C=1,class_weight=None,penalty='l2',solver='sag',max_iter=300)
        #clf2 = LinearSVC(class_weight='balanced',penalty='l1',dual=False)
        clf2 = LogisticRegression(C=1.,class_weight='balanced',penalty='l1',solver='liblinear',max_iter=300)
        #clf2 = LogisticRegression(C=1.,class_weight='balanced',penalty='l2',solver='sag',max_iter=300)
        #clf2 = SVC(C=1.,cache_size=500,kernel='linear',class_weight='balanced')
        #clf2 = RandomForestClassifier(n_estimators=20,class_weight='balanced')
        #param_grid = dict(C=(10**np.arange(1.,-2.,-0.5)))
        #param_grid = dict(C=(np.arange(3,1,-0.5)))
        #param_grid = dict(C=(np.logspace(-0.5, 2., 30)))
        #param_grid = dict(C=(np.logspace(1., -2., 15)))
        #if min_c>(10**-0.2):
        #    param_grid = dict(C=(np.logspace(np.log10(min_c), 1, 15)))
        #else:
        param_grid = dict(C=(np.logspace(-.2, 1, 15)))
        #param_grid = dict(C=(np.logspace(-.1, 0.5, 30)))
        #param_grid = dict(C=(np.logspace(0,0.00001, 2)))
        #param_grid = dict(C=(np.logspace(np.log10(min_c), 0., 15)))
        #param_grid = dict(C=(1,1.10001)) 
        #param_grid = dict(n_estimators=(20,10))

        # 2 levels balancing 
        '''
        new_classes = np.zeros_like(y)
        new_classes[(y==0) & (hm_y==0)]=0
        new_classes[(y==1) & (hm_y==0)]=1
        new_classes[(y==0) & (hm_y==1)]=2
        new_classes[(y==1) & (hm_y==1)]=3

        tmp_samp_w = len(new_classes) / (len(np.unique(new_classes))*1. * np.bincount(new_classes))
        tmp_samp_w = (1.*(tmp_samp_w/tmp_samp_w.sum()))
        sample_w = new_classes.copy().astype(float)
        sample_w[new_classes==0] = tmp_samp_w[0]
        sample_w[new_classes==1] = tmp_samp_w[1]
        sample_w[new_classes==2] = tmp_samp_w[2]
        sample_w[new_classes==3] = tmp_samp_w[3]
        '''
        new_classes = np.zeros_like(y)
        new_classes[(y==0) ]=0
        new_classes[(y==1) ]=1

        tmp_samp_w = len(new_classes) / (len(np.unique(new_classes))*1. * np.bincount(new_classes))
        #tmp_samp_w = (1.*(tmp_samp_w/tmp_samp_w.sum()))
        sample_w = new_classes.copy().astype(float)
        sample_w[new_classes==0] = tmp_samp_w[0]
        sample_w[new_classes==1] = tmp_samp_w[1]

        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(hm_y,n_folds=gs),fit_params=dict(sample_weight=sample_w), n_jobs=-1,scoring='accuracy')
        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(hm_y,n_folds=gs),fit_params=dict(sample_weight=proba), n_jobs=-1,scoring='accuracy')
        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(hm_y,n_folds=gs), n_jobs=-1,scoring='precision_weighted')
        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(hm_y,n_folds=gs), n_jobs=-1,scoring='accuracy')
        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedShuffleSplit(hm_y, n_iter=50, test_size=.2,random_state=1), n_jobs=-1,scoring='accuracy')#f1_weighted
        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedShuffleSplit(hm_y, n_iter=50, test_size=.2,random_state=1), n_jobs=-1,scoring='f1_weighted')
        gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedShuffleSplit(n_splits=50, test_size=.2,random_state=1), n_jobs=-1,scoring='precision_weighted')
        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedShuffleSplit(hm_y, n_iter=50, test_size=.2,random_state=1), n_jobs=-1,fit_params=dict(sample_weight=sample_w),scoring='precision_weighted')
        gridclf.fit(xwl2,hm_y)
        clf2 = gridclf.best_estimator_
        #clf2.fit(xw[train_index,:][:,idx_sz],hm_y)
        if self.verbose:
            print clf2
            print clf2.coef_

        self.clf2 = clf2

        #self.fit_2branch(xwl2,hm_y,y)
        #self.robust_coef(xwl2,hm_y)

    def fit_branchmodel(self,xwl2,hm_y):
        clf = LogisticRegression(C=1.,class_weight='balanced',penalty='l1',solver='liblinear',max_iter=300)
        param_grid = dict(C=(np.logspace(-.2, 1, 15)))
        #gridclf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(hm_y,n_folds=4), n_jobs=-1,scoring='precision_weighted')
        gridclf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedShuffleSplit(n_splits=50, test_size=.2,random_state=1), n_jobs=-1,scoring='accuracy')
        gridclf.fit(xwl2,hm_y)
        return gridclf.best_estimator_

    def fit_2branch(self,xwl2,hm_y,y_pred):
        mask_ = y_pred==1
        self.clf_0 = self.fit_branchmodel(xwl2[~mask_,:],hm_y[~mask_])
        self.clf_1 = self.fit_branchmodel(xwl2[mask_,:],hm_y[mask_])

    def predict_2branch(self,xwl2,y_pred):
        y_pred_l2 = np.zeros_like(y_pred).astype(float)
        mask_ = y_pred==1
        y_pred_l2[~mask_] = self.clf_0.decision_function(xwl2[~mask_,:])
        y_pred_l2[mask_] = self.clf_1.decision_function(xwl2[mask_,:])
        return y_pred_l2

    def robust_coef(self,xwl2,hm_y,n_iter=100):
        skf = StratifiedShuffleSplit(n_splits=n_iter, test_size=.2,random_state=1)
        coefs_ = []
        intercept_ = []
        for train,test in skf.split(xwl2,hm_y):
            self.clf2.fit(xwl2[train,:],hm_y[train])
            coefs_.append(self.clf2.coef_)
            intercept_.append(self.clf2.intercept_)
        self.clf2.coef_ = np.stack(coefs_).mean(0)
        self.clf2.intercept_ = np.stack(intercept_).mean(0)

    def predict(self,xw,xwl2):
        y_pred1 = self.clf1.decision_function(xw)
        y_pred2 = self.clf2.decision_function(xwl2)
        #y_pred2 = self.predict_2branch(xwl2,y_pred1)
        #y_pred2 = self.clf2.predict(xwl2)
        return np.array([y_pred1,y_pred2]).T

    def score(self,xw,xwl2,y):
        res = self.predict(xw,xwl2)
        l1_y_pred = res[:,0]
        risk_mask = res[:,1]>0
        right_cases = accuracy_score(y[risk_mask],res[risk_mask,0])
        left_cases = accuracy_score(y[~risk_mask],res[~risk_mask,0])

        #print 'clf3: ',self.clf3.score(xw,y)
        return accuracy_score(y,(l1_y_predi>0).astype(int)),left_cases,right_cases

    def suffle_hm(self,x,y,gamma=0.5,n_iter=50):
        hm_count = np.zeros_like(y).astype(float)
        hm = np.zeros_like(y).astype(float)
        skf = StratifiedShuffleSplit(n_splits=n_iter, test_size=.25,random_state=1)
        coefs_ = []
        sv_ = []
        for train,test in skf.split(x,y):
            self.clf1.fit(x[train,:],y[train])
            hm_count[test] += 1.
            hm[test] += (self.clf1.predict(x[test,:])==y[test]).astype(float)
            #coefs_.append(self.clf1.dual_coef_)
            #coefs_.append(self.clf1.coef_)
            #sv_.append(self.clf1.support_vectors_)
        proba = hm/hm_count
        if self.verbose:
            print(hm_count)
            print(proba)
        #self.clf1.dual_coef_ = np.stack(coefs_).mean(0)
        #self.clf1.support_vectors_ = np.stack(sv_).mean(0)
        #self.clf1.coef_ = np.stack(coefs_).mean(0)
        self.clf1.fit(x,y)
        return (proba>=gamma).astype(int),proba

