__author__ = 'Christian Dansereau'

import numpy as np
#from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import predlib as plib
from sklearn import preprocessing
from Proteus.proteus.matrix import tseries as ts
from Proteus.proteus.predic import betacluster as bc
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#data_path = '/home/cdansereau/Dropbox/McGill-publication/Papers/PredicAD/prediction_data_p2p/data_connec100_prediction.csv'
#data_path = '/home/cdansereau/Dropbox/McGill-publication/Papers/PredicAD/prediction_data_p2p/data_connec100_prediction_compcor.csv'
from sklearn import metrics
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold

def custom_scale(x):
    #print x
    #print np.mean(np.array(x),axis=0), np.std(np.array(x),axis=0)
    return (np.array(x) - np.array(x).mean(axis=0))/np.array(x).std(axis=0)

def estimate_std(p,n):
    print p,n,np.sqrt(p*(1-p)/n)
    return np.sqrt(p*(1-p)/n)

def estimate_unbalanced_std(y1,y2):
    if type(y1).__module__ != np.__name__:
        y1 = np.array(y1)
    if type(y2).__module__ != np.__name__:
        y2 = np.array(y2)

    idx_0 = np.where(y1==0)[0]
    idx_1 = np.where(y1==1)[0]

    p0 = metrics.accuracy_score(y1[idx_0],y2[idx_0])
    p1 = metrics.accuracy_score(y1[idx_1],y2[idx_1])

    #return 0.5*(estimate_std(p0,len(idx_0)) + estimate_std(p1,len(idx_1)))
    return 0.5*np.sqrt(estimate_std(p0,len(idx_0)) + estimate_std(p1,len(idx_1)))

class ConfoundsRm:

    def __init__(self, confounds, data,intercept=True):
        self.fit(data, confounds)

    def fit(self, confounds, data,intercept=True):
        if len(confounds) == 0:
            self.nconfounds = 0
        else:
            self.nconfounds = confounds.shape[1]
            self.reg = linear_model.LinearRegression(fit_intercept=intercept)
            self.reg.fit(data, confounds)

    def transform(self, confounds, data):
        # compute the residual error
        if self.nconfounds == True:
            return data
        else:
            return data - self.reg.predict(confounds)
    def nConfounds(self):
        return self.nconfouds

def compute_acc_noconf(x,y,verbose=False):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)

    # remove intra matrix mean and var
    #x = ts.normalize_data(x)
    #cv = cross_validation.KFold(len(y),n_folds=10)
    cv = StratifiedKFold(y=encoder.transform(y), n_folds=10)
    #cv = cross_validation.LeaveOneOut(data_all.shape[0])

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    total_test_score=[]
    clf_array = []
    bc_all = []

    for i, (train, test) in enumerate(cv):

        select_x = x.copy()
        #betacluster = bc.BetaCluster(select_x[train,:],encoder.transform(y[train]),n_cluster=200,samp_ratio=0.9,k_feature=0)
        #bc_all.append(betacluster)

        #train_x = betacluster.transform(select_x[train,:])
        #sel = VarianceThreshold()
        #train_x = sel.fit_transform(select_x[train,:])
        train_x = select_x[train,:]
        train_y = encoder.transform(y[train])
        #test_x = betacluster.transform(select_x[test,:])
        #test_x = sel.transform(select_x[test,:])
        test_x = select_x[test,:]
        test_y = encoder.transform(y[test])

        print train_x.shape
        clf = SVC(kernel='linear', class_weight='auto', C=.01)
        #clf = LinearSVC(class_weight='auto', C=0.1)
        clf, score = plib.grid_search(clf, train_x,train_y, n_folds=10, verbose=True, detailed=True)


        clf.fit(train_x,train_y)
        if verbose:
            #print('nSupport: ',clf.n_support_)
            print "Train:",clf.score(train_x,train_y)
            print "Test :",clf.score(test_x,test_y)
            print "Prediction :",clf.predict(test_x)
            print "Real Labels:",test_y

        total_test_score.append( clf.score(test_x,test_y))

        clf_array.append(clf)

    print 'mean', np.mean(total_test_score), 'std',  np.std(total_test_score)
    return {'mean': np.mean(total_test_score),'std': np.std(total_test_score),'data': total_test_score}


def compute_acc_conf(x,y,confounds,verbose=False,balanced=True,loo=False,optimize=True,C=.01):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)

    # remove intra matrix mean and var
    #x = ts.normalize_data(x)
    #cv = cross_validation.KFold(len(y),n_folds=10)
    if loo:
        cv = cross_validation.LeaveOneOut(len(y))
    else:
        cv = StratifiedKFold(y=encoder.transform(y), n_folds=10)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    total_test_score=[]
    y_pred=[]
    clf_array = []
    bc_all = []

    prec = []
    recall = []
    
    if len(np.unique(y))==1:
        print 'Unique class: 100%', np.sum(encoder.transform(y)==0)/len(y)
        return (1., 0.)
    
    for i, (train, test) in enumerate(cv):

        select_x = x.copy()

        crm = ConfoundsRm(confounds[train,:],select_x[train,:])

        #betacluster = bc.BetaCluster(crm.transform(confounds[train,:],select_x[train,:]),encoder.transform(y[train]),100,k_feature=200)
        #bc_all.append(betacluster)

        if balanced:
            clf = SVC(kernel='linear', class_weight='auto', C=C)
        else:
            clf = SVC(kernel='linear',C=C)
        #clf = NuSVC(kernel='linear', nu=.01)
        #clf = LogisticRegression(penalty='l2',C=100.)
        #clf = LDA(solver='lsqr', shrinkage='auto')
        #clf = LDA()

        #clf.probability = True
        if optimize:
            clf, score = plib.grid_search(clf, crm.transform(confounds[train,:],select_x[train,:]),encoder.transform(y[train]), n_folds=5, verbose=False)
        #print select_x.shape


        clf.fit(crm.transform(confounds[train,:],select_x[train,:]),encoder.transform(y[train]))
        total_test_score.append( clf.score(crm.transform(confounds[test,:],select_x[test,:]),encoder.transform(y[test])))
        clf_array.append(clf)

        prec.append(metrics.precision_score(encoder.transform(y[test]), clf.predict(crm.transform(confounds[test,:],select_x[test,:])) ))
        recall.append(metrics.recall_score(encoder.transform(y[test]), clf.predict(crm.transform(confounds[test,:],select_x[test,:])) ))

        if loo:
            y_pred.append(clf.predict(crm.transform(confounds[test,:],select_x[test,:])))
        if verbose:
            print('nSupport: ',clf.n_support_)
            print "Train:",clf.score(crm.transform(confounds[train,:],select_x[train,:]),encoder.transform(y[train]))
            print "Test :",clf.score(crm.transform(confounds[test,:],select_x[test,:]),encoder.transform(y[test]))
            print "Prediction :",clf.predict(crm.transform(confounds[test,:],select_x[test,:]))
            print "Real Labels:",encoder.transform(y[test])
            print('Precision:',prec[-1],'Recall:',recall[-1])

    if loo:
        total_std_test_score = estimate_std(metrics.accuracy_score(encoder.transform(y), np.array(y_pred)),len(y))
        print('Mean:', np.mean(total_test_score),'Std:', total_std_test_score,'AvgPrecision:',np.mean(prec),'AvgRecall:',np.mean(recall) )
        return (np.mean(total_test_score), total_std_test_score, len(y))
    else:
        print('Mean:', np.mean(total_test_score),'Std:', np.mean(total_std_test_score),'AvgPrecision:',np.mean(prec),'AvgRecall:',np.mean(recall) )
        return (np.mean(total_test_score), np.mean(total_std_test_score))

def sv_metric(n,nsv):
    return nsv/float(n)
    #return (n-nsv)/float(n) #lower the n sv is greater the score


def get_opt_model(x,y):

    # grid search and SVM
    clf = svm.SVC(kernel='rbf', class_weight='auto')
    clf.probability = True
    #clf = svm.SVC(kernel='rbf')
    clf, best_score = plib.grid_search(clf, x, y, n_folds=10, verbose=False)
    clf.fit(x,y)
    return clf

def basicconn(skf,X,y):
    total_score = 0
    for train_index, test_index in skf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        # Feature selection
        #selectf = SelectFpr().fit(X[train_index],y[train_index])
        #selectf = SelectKBest(f_classif, k=750).fit(X[train_index],y[train_index])
        #tmp_x = selectf.transform(X[train_index])
        # Train
        #clf = RandomForestClassifier(n_estimators=20)
        #clf = clf.fit(tmp_x, y[train_index])
        #clf.feature_importances_
        # SVM
        #clf = svm.LinearSVC()
        #clf = svm.SVC()
        #clf.fit(tmp_x, y[train_index])
        clf = plib.classif(X[train_index], y[train_index])
        #clf.support_vec()
        # Test
        #pred = clf.predict(selectf.transform(X[test_index]))
        pred = clf.predict(X[test_index])
        print "Target     : ", y[test_index]
        print "Prediction : ", pred
        matchs = np.equal(pred, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)


def splitconn(skf,X,y):
    total_score = 0
    for train_index, test_index in skf:
        # Train
        clf1 = plib.classif(X[train_index, 0:2475:1], y[train_index])
        clf2 = plib.classif(X[train_index, 2475:4950:1], y[train_index])
        pred1 = clf1.decision_function(X[train_index, 0:2475:1])
        pred2 = clf2.decision_function(X[train_index, 2475:4950:1])
        clf3 = svm.SVC()
        y[train_index].shape
        np.array([pred1, pred2])
        clf3.fit(np.array([pred1, pred2]).transpose(), y[train_index])
        #clf3 = plib.classif(np.matrix([pred1,pred2]).transpose(),y[train_index])

        # Test
        pred1 = clf1.decision_function(X[test_index, 0:2475:1])
        pred2 = clf2.decision_function(X[test_index, 2475:4950:1])
        predfinal = clf3.predict(np.matrix([pred1, pred2]).transpose())
        print "Target     : ", y[test_index]
        print "Prediction : ", predfinal
        matchs = np.equal(predfinal, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)


def multisplit(skf,X,y,stepsize=1000):
    total_score = 0
    for train_index, test_index in skf:
        wl = []
        pred1 = np.matrix([])
        # Training
        for x in range(0, len(X[0]), stepsize):
            clf1 = plib.classif(X[train_index, x:x + stepsize], y[train_index])
            tmp_p = np.matrix(clf1.decision_function(X[train_index, x:x + stepsize]))
            if pred1.size == 0:
                pred1 = tmp_p
            else:
                pred1 = np.concatenate((pred1, tmp_p), axis=1)
            wl.append(clf1)
        #selectf = SelectKBest(f_classif, k=5).fit(pred1, y[train_index])
        selectf = SelectFpr().fit(pred1, y[train_index])
        clf3 = AdaBoostClassifier(n_estimators=100)
        #clf3 = svm.SVC(class_weight='auto')
        #clf3 = RandomForestClassifier(n_estimators=20)
        clf3.fit(selectf.transform(pred1), y[train_index])
        # Testing
        predtest = np.matrix([])
        k = 0
        for x in range(0, len(X[0]), stepsize):
            tmp_p = np.matrix(wl[k].decision_function(X[test_index, x:x + stepsize]))
            if predtest.size == 0:
                predtest = tmp_p
            else:
                predtest = np.concatenate((predtest, tmp_p), axis=1)
            k += 1
        # Final prediction
        predfinal = clf3.predict(selectf.transform(predtest))
        print "Target     : ", y[test_index]
        print "Prediction : ", predfinal
        matchs = np.equal(predfinal, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)

