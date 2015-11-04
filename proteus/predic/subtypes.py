__author__ = 'Christian Dansereau'

import numpy as np
from sklearn.cluster import KMeans
from proteus.predic import clustering as cls
from proteus.matrix import tseries as ts
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MeanShift
from sklearn.lda import LDA
from sklearn.neighbors.nearest_centroid import NearestCentroid

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

    def assigneSubtype(self,nets,valid_cluster, valid_net_idx):
        classes = []
        dist_centroid = np.array([])
        for i in range(len(valid_net_idx)):
            classes.append(valid_cluster[i].predict(nets[:,valid_net_idx[i]])[0])
            #points = np.vstack((nets[:,valid_net_idx[i]],valid_cluster[i].cluster_centers_))
            #dist_ = squareform(pdist(points, metric='euclidean'))[0,1:]
            #classes.append(np.argmin(dist_))
            points = np.vstack((nets[:,valid_net_idx[i]],valid_cluster[i].cluster_centers_))
            dist_ = squareform(pdist(points, metric='euclidean'))[0,1:]
            dist_centroid = np.hstack((dist_centroid,dist_))
        
        return classes, dist_centroid

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

    def fit(self,net_data,nnet_cluster='auto',nSubtypes=3):
        self.nnet_cluster = nnet_cluster
        self.nSubtypes = nSubtypes
        
        if nnet_cluster == 'auto':
            #self.nnet_cluster = self.getClusters(net_data)
            self.valid_cluster, self.valid_net_idx = self.get_match_network(net_data,nnet_cluster,algo='meanshift')
        else:
            self.valid_cluster, self.valid_net_idx = self.get_match_network(net_data,nnet_cluster,algo='kmeans')

        #self.valid_cluster = self.clust_list
        #self.valid_net_idx = range(len(self.valid_cluster))
        self.assign_net = np.array([])
        self.dist_net   = np.array([])
        for i in range(net_data.shape[0]):
            if i == 0 :
                classes_, dist_ = self.assigneSubtype(net_data[i,:,:],self.valid_cluster, self.valid_net_idx)
                self.dist_net = dist_
                self.assign_net = classes_
            else:
                classes_, dist_ = self.assigneSubtype(net_data[i,:,:],self.valid_cluster, self.valid_net_idx)
                self.dist_net = np.vstack((self.dist_net,dist_))
                self.assign_net = np.vstack((self.assign_net,classes_))

        # group subjects with the most network classifing them together
        # compute the consensus clustering
        self.consensus = cls.hclustering(self.assign_net,self.nSubtypes)
        # save the centroids in a method
        self.clf_subtypes = NearestCentroid()
        self.clf_subtypes.fit(self.assign_net,self.consensus)
        self.consensus = self.clf_subtypes.predict(self.assign_net)
        #print "score: ", self.clf_subtypes.score(self.assign_net,self.consensus)

        return self.consensus


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
