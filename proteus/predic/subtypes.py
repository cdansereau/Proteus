__author__ = 'Christian Dansereau'

import numpy as np
from sklearn.cluster import KMeans
from proteus.predic import clustering as cls
from proteus.matrix import tseries as ts
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MeanShift
from sklearn.lda import LDA
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import preprocessing

class clusteringST:
    '''
    Identification of sub-types for prediction
    ''' 

    def getClusters(self,net_data):
        self.avg_bin_mat = np.zeros((net_data.shape[0],net_data.shape[0]))
        self.avg_n_clusters = 0
        self.clust_list = []
        for i in range(net_data.shape[2]):
            ms = MeanShift()
            ms.fit(net_data[:,:,i])
            self.clust_list.append(ms)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            n_clusters_ = len(np.unique(labels))
            #print(labels,cluster_centers.shape,n_clusters_)
            #bin_mat = np.zeros(avg_bin_mat.shape)
             
            bin_mat = cls.ind2matrix(labels+1)>0
            self.avg_bin_mat += bin_mat
            self.avg_n_clusters += n_clusters_
    
        self.avg_bin_mat /= net_data.shape[2]
        self.avg_n_clusters /= net_data.shape[2]
        return self.avg_n_clusters
    
    def getMeanClustering(self):
        return self.avg_bin_mat

    def get_match_network(self,net_data,ncluster,algo='kmeans'):
        '''
        net_data: 3d volume (subjects x vecnetwork x vecnetwork)
        ncluster: number of groups to partition the subjects
        algo: (default: kmeans) kmeans, meanshift.
        '''
        valid_net_idx = []
        valid_cluster = []
        self.avg_bin_mat = np.zeros((net_data.shape[0],net_data.shape[0]))
        self.avg_n_clusters = 0
        
        for i in range(net_data.shape[2]):
            # Compute clustering with for each network
            if algo == 'kmeans':
                clust = KMeans(init='k-means++', n_clusters=ncluster, n_init=10)
            else:
                clust = MeanShift()
            
            #t0 = time.time()
            clust.fit(net_data[:,:,i])
            #t_batch = time.time() - t0
            # Compute the stability matrix among networks
            bin_mat = cls.ind2matrix(clust.labels_+1)>0
            self.avg_bin_mat += bin_mat
            self.avg_n_clusters += len(np.unique(clust.labels_))
            
            valid_cluster.append(clust)
            valid_net_idx.append(i)
        self.avg_bin_mat /= net_data.shape[2]
        self.avg_n_clusters /= net_data.shape[2]
        
        return valid_cluster, valid_net_idx


    def assigneDist(self,nets,valid_cluster, valid_net_idx):
        classes = np.array([])
        for i in range(len(valid_net_idx)):
            #print  np.hstack((classes,(valid_cluster[i].transform(nets[:,valid_net_idx[i]])[0])))
            points = np.vstack((nets[:,valid_net_idx[i]],valid_cluster[i].cluster_centers_))
            dist_ = squareform(pdist(points, metric='euclidean'))[0,1:]
            #dist_ = squareform(pdist(points, metric='correlation'))[0,1:]
            classes = np.hstack((classes,dist_))
            #classes.append(np.argmin(dist_))
        return classes

    def transform_low_scale_old(self,net_data):
        # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork
        nnet_cluster = np.max(self.ind_low_scale)
        net_data_low = []
        net_data_low = np.zeros((net_data.shape[0],nnet_cluster,net_data.shape[2]))

        for i in range(nnet_cluster):
            # average the apropriate parcels and scale them
            #net_data_low[:,i,:] = preprocessing.scale(net_data[:,self.ind_low_scale==i+1,:].mean(axis=1), axis=1)
            net_data_low[:,i,:] = net_data[:,self.ind_low_scale==i+1,:].mean(axis=1)
        return net_data_low

    def fit(self,net_data_low,nSubtypes=3,reshape_w=True):
        self.flag_2level = False
        self.nnet_cluster = net_data_low.shape[1]
        self.nSubtypes = nSubtypes

        #ind_low_scale = cls.get_ind_high2low(low_res_template,orig_template)
        #self.ind_low_scale = ind_low_scale

        # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork
        #net_data_low = transform_low_scale(ts_data,self.ind_low_scale)
        self.net_data_low = net_data_low

        # st_templates --> Dimensions: nNetwork_low, nSubtypes, nNetwork
        st_templates = []
        for i in range(net_data_low.shape[1]):
            # indentity matrix of the corelation between subjects
            #tmp_subj_identity = np.corrcoef(net_data_low[:,i,:])
            #ind_st = cls.hclustering(tmp_subj_identity,nSubtypes)
            # subjects X network_nodes
            #ind_st = cls.hclustering(net_data_low[:,i,:]-np.mean(net_data_low[:,i,:],axis=0),nSubtypes)
            ind_st = cls.hclustering(net_data_low[:,i,:],nSubtypes)

            for j in range(nSubtypes):
                if j == 0:
                    st_templates_tmp = net_data_low[:,i,:][ind_st==j+1,:].mean(axis=0)[np.newaxis,...]
                else:
                    st_templates_tmp = np.vstack((st_templates_tmp,net_data_low[:,i,:][ind_st==j+1,:].mean(axis=0)[np.newaxis,...]))

            if i == 0:
                st_templates = st_templates_tmp[np.newaxis,...]
            else:
                st_templates = np.vstack((st_templates,st_templates_tmp[np.newaxis,...]))

        self.st_templates = st_templates

        # calculate the weights for each subjects
        self.W =  self.compute_weights(net_data_low,self.st_templates)
        if reshape_w:
            return self.reshapeW(self.W)
        else:
            return self.W

    def fit_2level(self,net_data_low_l1,net_data_low_l2,nSubtypes_l1=5,nSubtypes_l2=2,reshape_w=True):
        self.flag_2level = True
        self.nnet_cluster = net_data_low_l1.shape[1]
        self.nSubtypes = nSubtypes_l1 * nSubtypes_l2
        self.nSubtypes_l1 = nSubtypes_l1
        self.nSubtypes_l2 = nSubtypes_l2

        # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork
        self.net_data_low = net_data_low_l1
        self.net_data_low_l2 = net_data_low_l2

        ####
        # LEVEL 1
        ####
        # st_templates --> Dimensions: nNetwork_low, nSubtypes, nNetwork
        st_templates = []
        for i in range(net_data_low_l1.shape[1]):
            # indentity matrix of the corelation between subjects
            ind_st = cls.hclustering(net_data_low_l1[:,i,:],nSubtypes_l1)

            for j in range(nSubtypes_l1):
                if j == 0:
                    st_templates_tmp = net_data_low_l1[:,i,:][ind_st==j+1,:].mean(axis=0)[np.newaxis,...]
                else:
                    st_templates_tmp = np.vstack((st_templates_tmp,net_data_low_l1[:,i,:][ind_st==j+1,:].mean(axis=0)[np.newaxis,...]))

            if i == 0:
                st_templates = st_templates_tmp[np.newaxis,...]
            else:
                st_templates = np.vstack((st_templates,st_templates_tmp[np.newaxis,...]))

        self.st_templates_l1 = st_templates

        # calculate the weights for each subjects
        # W --> Dimensions: nSubjects,nNetwork_low, nSubtypes
        net_data_low_l2_tmp = np.vstack((net_data_low_l1,net_data_low_l2))
        self.W_l1 =  self.compute_weights(net_data_low_l2_tmp,self.st_templates_l1)

        ####
        # LEVEL 2                                                                                                                                                                                            
        ####
        # st_templates --> Dimensions: nNetwork_low, nSubtypes, nNetwork
        st_templates = []
        #st_templates = self.st_templates_l1.copy()
        #st_templates = st_templates[:,:,np.newaxis,:]
        for i in range(net_data_low_l2.shape[1]):

            # Iterate on all the Level1 subtypes (normal variability subtypes)
            for k in range(self.st_templates_l1.shape[1]):
                # Find the L1 subtype
                max_w = np.max(self.W_l1[:,i,:],axis=1)
                mask_selected_subj = (self.W_l1[:,i,k] == max_w)
                template2substract = self.st_templates_l1[i,k,:]
                if np.sum(mask_selected_subj)<=3:
                    print('Less then 2 subjects for network: ' +str(i) +' level1 ST: ' + str(k))
                    for j in range(nSubtypes_l2):
                        if (k == 0) & (j==0):
                            st_templates_tmp = self.st_templates_l1[i,k,:][np.newaxis,...]
                        else:
                            st_templates_tmp = np.vstack((st_templates_tmp,self.st_templates_l1[i,k,:][np.newaxis,...]))

                else:
                    # indentity matrix of the corelation between subjects
                    ind_st = cls.hclustering(net_data_low_l2_tmp[:,i,:][mask_selected_subj,...]-template2substract,nSubtypes_l2)
                    #ind_st = cls.hclustering(net_data_low[:,i,:],nSubtypes)
                    if len(np.unique(ind_st))<nSubtypes_l2:
                        print('Clustering generated less class then asked nsubjects: '+ str(len(ind_st)) +' network: ' +str(i) +' level1 ST: ' + str(k))
                    #if (i==6) & (k==3):
                        #print ind_st
                    for j in range(nSubtypes_l2):
                        if (k == 0) & (j==0):
                            st_templates_tmp = (net_data_low_l2_tmp[:,i,:][mask_selected_subj,...][ind_st==j+1,:]-template2substract).mean(axis=0)[np.newaxis,...]
                        else:
                            st_templates_tmp = np.vstack((st_templates_tmp,(net_data_low_l2_tmp[:,i,:][mask_selected_subj,...][ind_st==j+1,:]-template2substract).mean(axis=0)[np.newaxis,...]))


            if i == 0:
                st_templates = st_templates_tmp[np.newaxis,...]
            else:
                print st_templates.shape, st_templates_tmp.shape
                st_templates = np.vstack((st_templates,st_templates_tmp[np.newaxis,...]))

        self.st_templates_l2 = st_templates

        # calculate the weights for each subjects
        self.W_l2 =  self.compute_weights(net_data_low_l2,self.st_templates_l2)
        if reshape_w:
            return self.reshapeW(self.W_l2)
        else:
            return self.W_l2


    def compute_weights(self,net_data_low,st_templates):
        # calculate the weights for each subjects
        W = np.zeros((net_data_low.shape[0],st_templates.shape[0],st_templates.shape[1]))
        for i in range(net_data_low.shape[0]):
            for j in range(st_templates.shape[0]):
                for k in range(st_templates.shape[1]):
                    # Demean
                    average_template = np.median(self.net_data_low[:,j,:],axis=0)
                    #average_template = self.st_templates[j,:,:].mean(axis=0)
                    dm_map = net_data_low[i,j,:] - average_template
                    dm_map = preprocessing.scale(dm_map)
                    st_dm_map = st_templates[j,k,:] - average_template
                    W[i,j,k] = np.corrcoef(st_dm_map,dm_map)[-1,0:-1]

        return W

    def compute_weights_l2(self,net_data_low):

        corrected_ndl = net_data_low.copy()
        W_l1 = self.compute_weights(net_data_low,self.st_templates_l1)

        # calculate the weights for each subjects
        for i in range(net_data_low.shape[1]):
            for k in range(self.st_templates_l1.shape[1]):
                # Find the L1 subtype
                max_w = np.max(W_l1[:,i,:],axis=1)
                mask_selected_subj = (W_l1[:,i,k] == max_w)
                corrected_ndl[mask_selected_subj,i,:] = corrected_ndl[mask_selected_subj,i,:] - self.st_templates_l1[i,k,:]

        return self.compute_weights(corrected_ndl,self.st_templates_l2)


    def transform(self,net_data_low,reshape_w=True):
        '''
            Calculate the weights for each sub-types previously computed
        '''
        # compute the low scale version of the data
        #net_data_low = transform_low_scale(ts_data,self.ind_low_scale)

        if self.flag_2level:
            # calculate the weights for each subjects
            #W = self.compute_weights(net_data_low,self.st_templates_l2)
            W = self.compute_weights_l2(net_data_low)
        else:
            # calculate the weights for each subjects
            W = self.compute_weights(net_data_low,self.st_templates)

        if reshape_w:
            return self.reshapeW(W)
        else:
            return W

    def reshapeW(self,W):
        # reshape the matrix from [subjects, Nsubtypes, weights] to [subjects, vector of weights]
        xw = W.reshape((W.shape[0], W.shape[1]*W.shape[2]))
        return xw

    def fit_dev(self,net_data,nnet_cluster='auto',nSubtypes=3):
        self.nnet_cluster = nnet_cluster
        self.nSubtypes = nSubtypes

        if nnet_cluster == 'auto':
            #self.nnet_cluster = self.getClusters(net_data)
            self.valid_cluster, self.valid_net_idx = self.get_match_network(net_data,nnet_cluster,algo='meanshift')
        else:
            self.valid_cluster, self.valid_net_idx = self.get_match_network(net_data,nnet_cluster,algo='kmeans')

        #self.valid_cluster = self.clust_list
        #self.valid_net_idx = range(len(self.valid_cluster))
        for i in range(net_data.shape[0]):
            if i == 0 :
                self.assign_net = self.assigneDist(net_data[i,:,:],self.valid_cluster, self.valid_net_idx)
            else:
                self.assign_net = np.vstack(((self.assign_net,self.assigneDist(net_data[i,:,:],self.valid_cluster, self.valid_net_idx))))
        print 'Size of the new data map: ',self.assign_net.shape
        # group subjects with the most network classifing them together
        # compute the consensus clustering
        self.consensus = cls.hclustering(self.assign_net,self.nSubtypes)
        # save the centroids in a method
        self.clf_subtypes = NearestCentroid()
        self.clf_subtypes.fit(self.assign_net,self.consensus)
        self.consensus = self.clf_subtypes.predict(self.assign_net)
        #print "score: ", self.clf_subtypes.score(self.assign_net,self.consensus)

        return self.consensus

def transform_low_scale(ts_data,ind_low_scale):
    # compute the connectivity for at template at a given resolution
    allsubj_lowxhigh_conn=[]
    for i in range(len(ts_data)):
        ind_data = ts_data[i]
        tmp_ts_array =[]
        max_id_scale = ind_low_scale.max()
        # compute the time series for the low scale
        for j in range(max_id_scale):
            if j==0:
                tmp_ts_array = np.mean(ind_data[ind_low_scale==j+1,:],axis=0)
            else:
                tmp_ts_array = np.vstack((tmp_ts_array,np.mean(ind_data[ind_low_scale==j+1,:],axis=0)))

        # calculation of the correlation between each timeseries
        allsubj_lowxhigh_conn.append(np.corrcoef(np.vstack((ind_data,tmp_ts_array)))[-max_id_scale:,:][:,:-max_id_scale])
    # reorder the dimensions
    allsubj_lowxhigh_conn = np.dstack(allsubj_lowxhigh_conn)
    net_data_low = np.swapaxes(np.swapaxes(allsubj_lowxhigh_conn,1,2),0,1)
    # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork
    return net_data_low

def reshape_netwise(data_scale):
    # Reshape with the following dim: nSubjects, nfeatures, nfeatures
    for i in range(0,data_scale.shape[0]):
        if i==0:
            all_subjmat = ts.vec2mat(data_scale[i,:])[np.newaxis,...]
        else:        
            all_subjmat = np.vstack((all_subjmat,ts.vec2mat(data_scale[i,:])[np.newaxis,...]))
            
    #print all_subjmat.shape
    return all_subjmat

def format_nets(data, select_idx=[]):
    list_data = []
    for n in range(0,len(data)):        
        #tranform in matrix format
        if len(select_idx) > 0:
            clean_data = data[n][select_idx,:]
        else:
            clean_data = data[n]
        tmp_mat = reshape_netwise(clean_data)
        for i in range(0,tmp_mat.shape[2]):
            select_x = tmp_mat[:,:,i]
            list_data.append(select_x)
    return list_data 

def convSubScale(net_data,indtoconv):
    new_data = np.zeros((net_data.shape[0],net_data.shape[1],np.max(indtoconv)))
    for i in range(np.max(indtoconv)):
        new_data[:,:,i] = np.mean(net_data[:,:,indtoconv==i+1],axis=2)
    return new_data
