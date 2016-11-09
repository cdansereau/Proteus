__author__ = 'Christian Dansereau'

import glob,os
import numpy as np
import nibabel as nib
import multiprocessing
from multiprocessing import Pool
from ..visu import progress
from ..predic import prediction

##### grab rmaps #####
def grab_dynamic(connectivity_path,seed_index,dynamic):
    #print connectivity_path
    data = np.load(connectivity_path)
    if dynamic:
        return [data['dynamic_data'][:,seed_index,:],data['avg_data'][seed_index,:]]
    else:
        return data['avg_data'][seed_index,:]

def search_path(path,subject_id):
    list_of_files = glob.glob(str(path)+'*'+str(subject_id)+'*dynamic.npz')
    return list_of_files

def grab_rmap(subject_list,path,seed_index,dynamic=False,flag_std=False):
    dynamic_data = []
    avg_data     = []
    pbar = progress.Progbar(len(subject_list))
    k=0
    for sid in subject_list:
        k+=1
        path_subj = search_path(path,sid)[0]
        if dynamic:
            tmp_data = grab_dynamic(path_subj,seed_index,dynamic)
            dynamic_data.append(tmp_data[0])
            avg_data.append(tmp_data[1])
        else:
            tmp_data = grab_dynamic(path_subj,seed_index,dynamic)
            avg_data.append(tmp_data)

        if flag_std:
            dynamic_std.append(tmp_data[2])
            std_ref.append(tmp_data[3])
        pbar.update(k)

    if flag_std:
        return [dynamic_data,np.stack(avg_data),np.vstack(dynamic_std),np.stack(std_ref)]
    else:
        if dynamic:
            return [dynamic_data,np.stack(avg_data)]
        else:
            return np.stack(avg_data)

##### make rmaps #####

def dynamic_rmaps(data_,partition,voxel_mask,window=20):
    data = data_.copy()
    cf_rm = prediction.ConfoundsRm(data_[voxel_mask].mean(0).reshape(-1,1),data_[voxel_mask].T,intercept=False)
    data[voxel_mask] = cf_rm.transform(data_[voxel_mask].mean(0).reshape(-1,1),data_[voxel_mask].T).T

    n_iter = int(data.shape[-1]/(window/2.))-1
    dynamic_data = []
    for widx in range(n_iter):
        dynamic_data.append(prediction.get_corrvox(data[...,widx*(window/2):(widx+1)*window],voxel_mask,partition))
    dynamic_data = np.stack(dynamic_data)
    avg_ref = prediction.get_corrvox(data,voxel_mask,partition)
    return dynamic_data,avg_ref


def dynamic_rmaps_std(data_,partition,voxel_mask,window=20):
    data = data_.copy()
    cf_rm = prediction.ConfoundsRm(data_[voxel_mask].mean(0).reshape(-1,1),data_[voxel_mask].T,intercept=False)
    data[voxel_mask] = cf_rm.transform(data_[voxel_mask].mean(0).reshape(-1,1),data_[voxel_mask].T).T

    n_iter = int(data.shape[-1]/(window/2.))-1
    dynamic_data = []
    for widx in range(n_iter):
        dynamic_data.append(prediction.get_corrvox_std(data[...,widx*(window/2):(widx+1)*window],voxel_mask,partition))
    dynamic_data = np.stack(dynamic_data)
    avg_ref = prediction.get_corrvox_std(data,voxel_mask,partition)
    return dynamic_data,avg_ref

def compute_seed_map(seed_partition,brain_mask,list_files,subject_ids,output_path,multiprocess=True):
    print('Compute seed maps ...')
    n_seed = len(np.unique(seed_partition))
    pbar = progress.Progbar(len(list_files))
    seed_list = []
    params = []
    for ii in range(len(list_files)):
        subj_id = str(subject_ids[ii])
        new_path = output_path+'fmri_'+subj_id+'_'+str(n_seed)+'_vox_gs_dynamic.npz'
        if multiprocess:
            params.append( (subj_id,list_files[ii],seed_partition,brain_mask,new_path))
        else:
            seed_map_multiprocess((subj_id,list_files[ii],seed_partition,brain_mask,new_path))
        seed_list.append(new_path)
        pbar.update(ii+1)
    if multiprocess:
        p = Pool(processes=multiprocessing.cpu_count()-1)
        results = p.map_async(seed_map_multiprocess, params)

    return seed_list

def seed_map_multiprocess((subj_id,file_path,seed_partition,brain_mask,output_path)):
    vol_file = nib.load(file_path).get_data()
    dynamic_data,avg_data = dynamic_rmaps(vol_file,seed_partition.get_data(),brain_mask)
    del vol_file
    np.savez_compressed(output_path,dynamic_data=dynamic_data,avg_data=avg_data)
    del dynamic_data,avg_data
    return new_path

