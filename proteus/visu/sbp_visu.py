import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import roc_curve, auc
from nistats import glm as nsglm


def plot_roc(y_test,y_score):

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()

def sbp_roc(sbp_dat):
    plt.figure(figsize=(6,6))
    #plt.subplot(2,2,1,colspan=2)
    plt.subplot2grid((3,2), (0,0), colspan=2,rowspan=2)
    plot_roc(sbp_dat[:,0],sbp_dat[:,1])
    plt.subplot2grid((3,2), (2,0))
    plot_roc(sbp_dat[sbp_dat[:,2]<=0,0],sbp_dat[sbp_dat[:,2]<=0,1])
    plt.subplot2grid((3,2), (2,1))
    plot_roc(sbp_dat[sbp_dat[:,2]>0,0],sbp_dat[sbp_dat[:,2]>0,1])
    plt.tight_layout()

def get_hm(clf,x,y):
    return np.array(clf.predict(x) == y,dtype=int)

def get_hm_labelonly(clf,x,y,label=1):
    # special HM calculation base only on a sub-group (denoted by label)
    hm_ = (y == label) & (clf.predict(x) == y)
    return np.array(hm_,dtype=int)
    
def get_hm_score_(lr,x,hm_y):

    w_coef = lr.coef_[0]

    df_data = lr.decision_function(x)
    print hm_y.shape
    print df_data.shape
    return np.array([hm_y[df_data<0], hm_y[df_data>0]])

def idx_decision(lr,data):
    return lr.decision_function(data)

def estimate_hitmiss(clf,x,y):
   
    label=1
    hm_results = []
    predictions =[]
    for i in range(len(y)):
        train_idx = np.array(np.hstack((np.arange(0,i),np.arange(i+1,len(y)))),dtype=int)
        #print train_idx.shape
        clf.fit(x[train_idx,:],y[train_idx])
        #print clf.predict(x[i,:]) == y[i]
        hm_results.append(int(clf.predict(x[i,:]) == y[i]))
        predictions.append(clf.predict(x[i,:]))
        #hm_results.append(int((y[i] == label) & (clf.predict(x[i,:]) == y[i]) ))#   clf.predict(x[i,:]) == y[i]))

    predictions = np.array(predictions)
    hm_results = np.array(hm_results)
    #print hm_results.shape
    clf.fit(x,y)
    return hm_results, predictions[:,0]
   
def show_hm_(lr,xtrain,hm_y,show_fig=True):

    w_coef = lr.coef_[0]

    idx_coef = np.argsort(w_coef)
    #print w_coef[idx_coef]

    lr_curve = [1./(1.+np.exp(-i)) for i in np.arange(-3.,3,0.1)]

    df_data = lr.decision_function(xtrain)
    
    if show_fig==True:
        plt.figure()
        y_tmp = lr.predict(xtrain)[hm_y==1]
        plt.plot(df_data[hm_y==1],y_tmp+np.random.normal(0, 0.1, size=len(y_tmp)),'ro')
        y_tmp = lr.predict(xtrain)[hm_y<=0]
        plt.plot(df_data[hm_y<=0],y_tmp+np.random.normal(0, 0.1, size=len(y_tmp)),'bo')

        # acc of the left side
        print (np.mean(hm_y[df_data<0]))
        # acc of the right side
        print (np.mean(hm_y[df_data>0]))
        print('Size',len(hm_y[df_data<0]),len(hm_y[df_data>0]))

        plt.plot(np.arange(-3.,3,0.1),lr_curve)
        plt.xlabel('Decision function\n ACC Left:' + "{0:.2f}".format(100*np.mean(hm_y[df_data<0]))+'%' + ' ACC Right:' + "{0:.2f}".format(100*np.mean(hm_y[df_data>0]))+'%')
        plt.ylabel('Prediction of the logistic regression')

        #plt.legend(['SZ','CTRL'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.legend(['Hit','Miss','logistic regression curve'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(['Hit','Miss','logistic regression curve'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
        plt.axis([-3, 3, -0.2, 1.5])
    return np.array([np.mean(hm_y[df_data<0]), np.mean(hm_y[df_data>0])])

def show_explicit_hm_(df_data,hm_y,labels=['Hit','Miss','logistic regression curve']):
    #sns.despine()
    sns.set_style("white")
    flatui = [ "#e74c3c","#3498db","#34495e", "#95a5a6", "#9b59b6", "#2ecc71"]
    sns.set_palette(flatui)
    
    lateral = df_data>0

    lr_curve = [1./(1.+np.exp(-i)) for i in np.arange(-3.,3,0.1)]
    
    randst = np.random.RandomState(seed=0)
    
    fig = plt.figure(dpi=250)
    y_tmp = lateral[hm_y==1]
    plt.plot(df_data[hm_y==1],y_tmp+randst.normal(0, 0.1, size=len(y_tmp)),'o')
    y_tmp = lateral[hm_y<=0]
    plt.plot(df_data[hm_y<=0],y_tmp+randst.normal(0, 0.1, size=len(y_tmp)),'o')


    # acc of the left side
    print (np.mean(hm_y[df_data<0]))
    # acc of the right side
    print (np.mean(hm_y[df_data>0]))
    print('Size',len(hm_y[df_data<0]),len(hm_y[df_data>0]))

    plt.plot(np.arange(-3.,3,0.1),lr_curve)
    plt.xlabel('Decision function ACC Left:' + "{0:.2f}".format(100*np.mean(hm_y[df_data<0]))+'%' + ' ACC Right:' + "{0:.2f}".format(100*np.mean(hm_y[df_data>0]))+'%')
    plt.ylabel('Prediction of the logistic regression')

    #plt.legend(['Hit','Miss','logistic regression curve'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(labels,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    plt.axis([-3, 3, -0.2, 1.5])
    return np.array([np.mean(hm_y[df_data<0]), np.mean(hm_y[df_data>0])])



def lr_(xtrain,hm_y,crange=[-.5,3]):
    lr = LogisticRegression(C=1e5,class_weight='balanced')#SVC(kernel='linear', class_weight='balanced', C=1)
    param_grid = dict(C=(10.**np.arange(crange[0],crange[1],0.01)))
    grid_lr = GridSearchCV(lr, param_grid=param_grid, cv=10, n_jobs=-1)
    grid_lr.fit(xtrain,hm_y)

    lr = grid_lr.best_estimator_
    print xtrain.shape, hm_y.shape

    print lr.score(xtrain,hm_y)
    print lr
    df_data = lr.decision_function(xtrain)
    #hm_y[df_data<0]
    return df_data<0,lr, xtrain[df_data<0,:], hm_y[df_data<0]


def nest_lr_(xtrain,ytrain):
    
    # SVM classifier
    #clf = LinearSVC(class_weight='balanced', C=1,penalty='l1',dual=False)
    clf = SVC(kernel='linear', class_weight='balanced', C=1,probability=False)
    #param_grid = dict(C=(10.**np.arange(-1,-2,-0.05)))#2.**np.arange(1,-5,-0.01))
    param_grid = dict(C=(10.**np.arange(0.5,-1,-0.05)))
    clf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(ytrain,n_folds=10), n_jobs=-1)
    clf.fit(xtrain,ytrain)
    
    clf = clf.best_estimator_
    print clf
    
    hm_y = estimate_hitmiss(clf,xtrain,ytrain)
    #hm_y = np.array(clf.predict(xtrain) == ytrain,dtype=int)
    
    # logistic regression
    lr = LogisticRegression(C=1e5,class_weight='balanced')
    #lr = SVC(kernel='linear', class_weight='balanced', C=1)
    #param_grid = dict(C=(10.**np.arange(.5,-1.5,-0.01)))
    param_grid = dict(C=(10.**np.arange(0.3,-0.5,-0.01)))# -0.2,-0.5
    #param_grid = dict(C=(10.**np.arange(0.5,-3,-0.01)))
    #param_grid = dict(C=(10.**np.arange(1,0,-0.01)))
    #StratifiedKFold(hm_y,n_folds=10)
    grid_lr = GridSearchCV(lr, param_grid=param_grid, cv=LeaveOneOut(len(hm_y)) , n_jobs=-1 )
    grid_lr.fit(xtrain,hm_y)

    lr = grid_lr.best_estimator_
    print xtrain.shape, hm_y.shape

    print lr.score(xtrain,hm_y)
    print lr
    df_data = lr.decision_function(xtrain)
    #hm_y[df_data<0]
    return clf,lr,hm_y

def nest_lr_svm(xtrain,xsvm,ytrain,gridsearch=True):
    
    # SVM classifier
    #clf = LinearSVC(class_weight='balanced', C=1,penalty='l1',dual=False)
    clf = SVC(kernel='linear', class_weight='balanced', C=10**-0.1,probability=False)
    if gridsearch:
        param_grid = dict(C=(10.**np.arange(0.7,0,-0.1)))#0.7,0,-0.1#-0.1,-1.1,-0.1
        #param_grid = dict(C=np.array([0.1,1,100,250,500,750,1000,1500,5000,10000]))
        clf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(ytrain,n_folds=10), n_jobs=-1,scoring='f1')
        clf.fit(xsvm,ytrain)
        print '\n'
        print clf.grid_scores_
        print '\n'
        clf = clf.best_estimator_
        
    print clf
    hm_y = estimate_hitmiss(clf,xsvm,ytrain)
    clf.fit(xsvm,ytrain)
    #hm_y = np.array(clf.predict(xsvm) == ytrain,dtype=int)
    print len(np.unique(hm_y))
    print np.unique(hm_y)
    # logistic regression
    lr = LogisticRegression(C=10**0.1,class_weight='balanced')
    #lr = SVC(kernel='linear', class_weight='balanced', C=1)
    
    if gridsearch:
        #param_grid = dict(C=(10.**np.arange(1.5,-0.25,-0.01)))
        param_grid = dict(C=(10.**np.arange(0.25,-0.25,-0.05)))#0.1,-0.25,-0.05
        #param_grid = dict(C=(10.**np.arange(0.3,-0.01,-0.01)))
        #StratifiedKFold(hm_y,n_folds=10)
        grid_lr = GridSearchCV(lr, param_grid=param_grid, cv=LeaveOneOut(len(hm_y)) , n_jobs=-1,scoring='f1')
        grid_lr.fit(xtrain,hm_y)
        lr = grid_lr.best_estimator_
    else:
        lr.fit(xtrain,hm_y)
    #print xtrain.shape, hm_y.shape
    #print lr.score(xtrain,hm_y)
    print lr
    df_data = lr.decision_function(xtrain)
    #hm_y[df_data<0]
    return clf,lr,hm_y


def score(net_data_low_main,y,confounds,svm_model,lr_model,cfrm_model,st_model):
    # remove confounds
    net_data_low_tmp = cfrm_model.transform(confounds,net_data_low_main.reshape((net_data_low_main.shape[0],net_data_low_main.shape[1]*net_data_low_main.shape[2])))
    net_data_low = net_data_low_tmp.reshape((net_data_low_tmp.shape[0],net_data_low_main.shape[1],net_data_low_main.shape[2]))
    
    # subtypes extraction
    x = st_model.transform(net_data_low)
    x_svm = net_data_low.reshape(net_data_low.shape[0],net_data_low.shape[1]*net_data_low.shape[2])
    
    #svm_model.score(x_svm,y)
    df_data = lr_model.decision_function(x)
    lateralization = lr_model.predict(x)
    hm_y = get_hm(svm_model,x_svm,y)
    
    show_explicit_hm_(lateralization,df_data,hm_y)
    
def score_w(net_data_low_main,y,confounds,lr1_model,lr2_model,cfrm_model,st_model,idx_sz):
    net_data_low_tmp = cfrm_model.transform(confounds,net_data_low_main.reshape((net_data_low_main.shape[0],net_data_low_main.shape[1]*net_data_low_main.shape[2])))
    net_data_low = net_data_low_tmp.reshape((net_data_low_tmp.shape[0],net_data_low_main.shape[1],net_data_low_main.shape[2]))
    
    # subtypes extraction
    x = st_model.transform(net_data_low)[:,idx_sz]
    #x_svm = net_data_low.reshape(net_data_low.shape[0],net_data_low.shape[1]*net_data_low.shape[2])
    
    #svm_model.score(x_svm,y)
    df_data = lr2_model.decision_function(x)
    lateralization = lr2_model.predict(x)
    hm_y = get_hm(lr1_model,x,y)
    
    show_explicit_hm_(df_data,hm_y)
    
from sklearn import metrics
def results_row(y_ref,y_pred,lr_decision,pos_label=1,average='weighted'):
    row=[]
    row.append(metrics.accuracy_score(y_ref,y_pred))
    row.append(metrics.precision_recall_fscore_support(y_ref,y_pred,pos_label=pos_label,average=average)[:3])
    row.append(metrics.accuracy_score(y_ref[lr_decision<0],y_pred[lr_decision<0]))
    row.append(metrics.precision_recall_fscore_support(y_ref[lr_decision<0], y_pred[lr_decision<0],pos_label=pos_label,average=average)[:3])
    row.append(metrics.accuracy_score(y_ref[lr_decision>0],y_pred[lr_decision>0]))
    row.append(metrics.precision_recall_fscore_support(y_ref[lr_decision>0], y_pred[lr_decision>0],pos_label=pos_label,average=average)[:3])
    return np.hstack(row)

def classif_repo(y_ref,y_pred,lr_decision):

    print '##################################################################'
    print 'Main'
    print classification_report(y_ref, y_pred)
    print 'ACC: '+str(accuracy_score(y_ref,y_pred))
    print 'Right:'
    print classification_report(y_ref[lr_decision>0], y_pred[lr_decision>0])
    print 'ACC: '+str(accuracy_score(y_ref[lr_decision>0],y_pred[lr_decision>0]))
    print 'Left:'
    print classification_report(y_ref[lr_decision<0], y_pred[lr_decision<0])
    print 'ACC: '+str(accuracy_score(y_ref[lr_decision<0],y_pred[lr_decision<0]))
    print '##################################################################'
    
def classif_repo_bimode(y_ref,y_pred,lr_decision_sz,lr_decision_ctrl):

    print '##################################################################'
    print 'Main'
    print classification_report(y_ref, y_pred)
    print 'ACC: '+str(accuracy_score(y_ref,y_pred))
    print '#####################'
    print 'Right SZ:'
    print classification_report(y_ref[y_pred==1][lr_decision_sz[y_pred==1]>0], y_pred[y_pred==1][lr_decision_sz[y_pred==1]>0])
    print 'ACC: '+str(accuracy_score(y_ref[y_pred==1][lr_decision_sz[y_pred==1]>0],y_pred[y_pred==1][lr_decision_sz[y_pred==1]>0]))
    print 'Left SZ:'
    print classification_report(y_ref[y_pred==1][lr_decision_sz[y_pred==1]<=0], y_pred[y_pred==1][lr_decision_sz[y_pred==1]<=0])
    print 'ACC: '+str(accuracy_score(y_ref[y_pred==1][lr_decision_sz[y_pred==1]<=0],y_pred[y_pred==1][lr_decision_sz[y_pred==1]<=0]))
    print '#####################'
    print 'Right CTRL:'
    print classification_report(y_ref[y_pred==0][lr_decision_ctrl[y_pred==0]>0], y_pred[y_pred==0][lr_decision_ctrl[y_pred==0]>0])
    print 'ACC: '+str(accuracy_score(y_ref[y_pred==0][lr_decision_ctrl[y_pred==0]>0],y_pred[y_pred==0][lr_decision_ctrl[y_pred==0]>0]))
    print 'Left CTRL:'
    print classification_report(y_ref[y_pred==0][lr_decision_ctrl[y_pred==0]<=0], y_pred[y_pred==0][lr_decision_ctrl[y_pred==0]<=0])
    print 'ACC: '+str(accuracy_score(y_ref[y_pred==0][lr_decision_ctrl[y_pred==0]<=0],y_pred[y_pred==0][lr_decision_ctrl[y_pred==0]<=0]))
    print '##################################################################'
    

def association_glm(var1,var2):
    # we assume that the var1 is a numerical variable already encoded
    
    
    #Encode variables
    x = var2
    
    #Sanity check
    if (x==0).sum()!=x.shape[0]:
    
        #Normalize
        x = (x - x.mean())/x.std()
        y = (var1 - var1.mean())/var1.std()
        #print y.shape,x.shape
        #print x
        #GLM
        contrast = [0,1]
        x_ = np.vstack((np.ones_like(x),x)).T
        labels, regression_result  = nsglm.session_glm(y[:,np.newaxis],x_)
        cont_results = nsglm.compute_contrast(labels,regression_result, contrast,contrast_type='t')
        pval = cont_results.p_value()[0]
        return cont_results.stat()[0],cont_results.p_value()[0][0][0]
    
    
    else:
        print '### Error nothing to regress ###'
        return np.NAN,np.NAN
    

def plot_st_w(stw,contrast):
    n_plot = stw.shape[1]
    
    n_x = np.ceil(np.sqrt(n_plot))
    n_y = np.ceil(n_plot/(n_x*1.))
    plt.figure(figsize=(1.5*n_x,2*n_y))
    f = plt.gcf()
    
    f.subplots_adjust(hspace=0.5)
    f.subplots_adjust(wspace=0.5)
    for i in range(n_plot):
        
        plt.subplot(1*n_y,n_x,i+1)
        ax = plt.gca()
        #plt.scatter(contrast,stw[:,i],color='k', alpha=0.5)
        data = [stw[:,i][contrast==j] for j in np.unique(contrast)]
        
        # Association test
        pval = association_glm(stw[:,i],contrast)[1]
        #print data
        #sns.violinplot(data)
        violin_parts = plt.violinplot(data, np.unique(contrast), points=15, widths=0.5,
                      showmeans=False, showextrema=True, showmedians=True)
        
        for pc in violin_parts['bodies']:
            pc.set_color('black')
            pc.set_facecolor('black')
            pc.set_edgecolor('black')
        violin_parts['cbars'].set_color('black')
        violin_parts['cmedians'].set_color('black')
        violin_parts['cmaxes'].set_color('black')
        violin_parts['cmins'].set_color('black')
        #violin_parts['cmeans'].set_color('black')
        
        plt.xticks(np.unique(contrast), ['HC','Patho'])
        plt.ylim([-1,1])
        plt.yticks(np.arange(-1, 1.1, 0.5))
        #ax.yaxis.grid(True,'major')
        ax.xaxis.grid(False)
        
        #ax.set_yticks([-0.5, 0.5, 0.5], minor=True)
        if pval<0.05:
            ax.set_title('* '+str(i+1))
        else:
            ax.set_title(str(i+1))
        if (i%n_x)>0:
            ax.minorticks_on
            ax.get_yaxis().set_visible(False)
            ax.get_yaxis().set_ticks([-0.5,0,0.5],minor=False)
            ax.yaxis.grid(True,which='major')
            
        else:
            plt.ylabel('Weights')
            ax.get_yaxis().set_visible(True)

        
