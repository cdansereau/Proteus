__author__ = 'Christian Dansereau'

import numpy as np
import math
import copy
from numba import jit
import math

@jit
def vec2map(v,partition):
    new_map = copy.deepcopy(partition)
    im = new_map.get_data()

    part = partition.get_data()
    for i in range(len(v)):
        idxs = np.where(part==(i+1))
        im[idxs] = v[i]
    return new_map

def normalize_data(x):
    x1 = x.copy()
    x1 = (x1 - x1.mean(axis=1)[0])#/x1.std(axis=1)[0]
    return x1

@jit
def vol2vec(vol):
    a = vol.shape
    vec_vol = np.reshape(vol,a[0]*a[1]*a[2])
    return vec_vol

@jit
def mat2vec(m,include_diag=False):
    # Hack to be compatible with matlab column-wise instead of row-wise
    if include_diag:
        inddown = np.triu_indices_from(m,0)
    else:
        inddown = np.triu_indices_from(m,1)

    inddown = (inddown[1], inddown[0])
    return m[inddown]

@jit
def vec2mat(vec, val_diag=1,include_diag=False):
    if vec.ndim > 1:
        vec = vec[:,0]
    M = len(vec)
    if include_diag:
        # Create the matrix with diagonal
        N = int(round(-1+(math.sqrt(1+8*M))/2))
        m = np.zeros((N,N))
        inddown = np.triu_indices_from(m,0)
    else:
        N = int(round((1+math.sqrt(1+8*M))/2))
        m = np.identity(N)*val_diag
        inddown = np.triu_indices_from(m,1)
    #indup = np.triu_indices_from(m,1)
    # need a hack to be compatible with matlab
    # python give indices row-wise and matlab column-wise ...
    inddown = (inddown[1], inddown[0])
    m[inddown] = vec
    m = m.T
    m[inddown] = vec
    return m

def ts2vol(vec,part):
    pass

@jit
def vec2vol(vec,part):
    vol = np.zeros(part.shape)
    for idx in range(0,len(vec)):
        idxs = np.where(part==(idx+1))
        vol[idxs] = vec[idx]

    return vol

@jit
def get_ts(vol,part):
    # create a NxT (partitions x time points)
    part = np.array(part,dtype=int)
    idx = np.unique(part)
    idx = idx[idx>0] #exclude the index 0
    ts=np.array([])
    for i in idx:
        mask_parcel = part == i
        # compute the average of the time series defined in a partition
        #print vol.shape,mask_parcel.shape
        #if len(vol[mask_parcel].shape)>1:
        ts_new = np.array(vol[mask_parcel].mean(axis=0))
        #else:
        #    ts_new = np.mean(vol[mask_parcel])
        #print ts_new.shape
        #print ts.shape
        if len(ts)==0:
            ts = np.vstack((ts_new,))
        else:
            ts = np.vstack((ts,ts_new))
    return ts


def get_connectome(vol,part):
    # convert the vol in time series
    ts = get_ts(vol,part)
    return np.corrcoef(ts)

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]

def test_ismember():
    a = np.array([[1,2,3,3,9,8]])
    b = np.array([[2,3,3,8]])
    res = ismember(a,b)
    assert np.all(new_a == a)

def test_mat2vec():
    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    va = mat2vec(a)
    # [2, 3, 4]
    assert np.all(va == [2, 3, 4])
    new_a = vec2mat(va)
    assert np.all(new_a == a)


def test_vec2mat():
    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    # [2, 3, 4]
    va == [2, 3, 4]
    new_a = vec2mat(va)
    assert np.all(new_a == a)

def vp(x,y=1):
    # z=vp(x,y); z = 3d cross product of x and y
    # vp(x) is the 3d cross product matrix : vp(x)*y=vp(x,y).
    #
    # by Giampiero Campa.  
    z=np.array([[  0,     -x[2],   x[1]],
                [  x[2],   0,     -x[0]],
                [ -x[1],   x[0],   0]])

    z=np.dot(z,y)
    return z

def transf2param(transf):

    if len(transf.shape)>2:
        N = transf.shape[2]
        rot = np.zeros([3, N])
        tsl = np.zeros([3, N])
        for num_n in range(N):
            [rot[:,num_n],tsl[:,num_n]] = transf2param(transf[:,:,num_n])
        return rot,tsl

    O=transf[0:3,3]
    R=transf[0:3,0:3]
    d=np.round(R[:,0][2]*1e12)/1e12

    if d==1:
        y=math.atan2(R[:,1][1],R[:,1][0])
        p=-np.pi/2
        r=-np.pi/2

    elif d==-1:
        y=math.atan2(R[:,1][1],R[:,1][0])
        p=np.pi/2
        r=np.pi/2

    else:
        sg=vp(np.array([0, 0, 1]).T,R[:,0])
        j2=sg/np.sqrt(np.dot(sg.T,sg))
        k2=vp(R[:,0],j2)

        r=math.atan2(np.dot(k2.T,R[:,1]),np.dot(j2.T,R[:,1]))
        p=math.atan2(-R[:,0][2],k2[2])
        y=math.atan2(-j2[0],j2[1])


    y1=y+(1-np.sign(y)-np.sign(y)**2)*np.pi
    p1=p+(1-np.sign(p)-np.sign(p)**2)*np.pi
    r1=r+(1-np.sign(r)-np.sign(r)**2)*np.pi

    # takes smaller values of angles
    if np.linalg.norm([y1, p1, r1]) < np.linalg.norm([y, p, r]):
        rot = np.array([r1,-p1,y1])
    else:
        rot = np.array([r,p,y])

    rot = (rot/np.pi)*180 # Conversion in degrees
    tsl = O
    return rot,tsl

def volterra(rot,tsl):
    rot_dev = np.hstack((np.array([0,0,0])[...,np.newaxis],rot[:,1:]-rot[:,:-1]))
    tsl_dev = np.hstack((np.array([0,0,0])[...,np.newaxis],tsl[:,1:]-tsl[:,:-1]))
    return np.vstack((rot,tsl,rot**2,tsl**2,rot_dev,tsl_dev,rot_dev**2,tsl_dev**2))

