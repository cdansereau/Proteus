__author__ = 'Christian Dansereau'

'''
Tools for registration of 3D volumes
'''

import numpy as np
import numpy.linalg as npl
from scipy import ndimage
from scipy.optimize import fmin_powell
from scipy.ndimage.filters import gaussian_filter


def resample_trans(sv, sv2sw_affine, tv2tw_affine, tv_shape, sw2tw_affine=np.eye(4)):
    # transform
    # start = time.time()
    transform_affine = npl.inv(np.dot(npl.inv(tv2tw_affine), np.dot(sw2tw_affine, sv2sw_affine)))

    # Split an homogeneous transform into its matrix and vector components.
    # The transformation must be represented in homogeneous coordinates.
    # It is split into its linear transformation matrix and translation vector
    # components.

    ndimin = transform_affine.shape[0] - 1
    ndimout = transform_affine.shape[1] - 1
    matrix = transform_affine[0:ndimin, 0:ndimout]
    vector = transform_affine[0:ndimin, ndimout]
    # print matrix,vector

    # interpolation
    new_volume = ndimage.affine_transform(sv, matrix,
                                          offset=vector,
                                          output_shape=tv_shape,
                                          order=1)
    # print(time.time() - start)
    return new_volume


def aff_tsf(xt, yt, zt, xr, yr, zr):
    transf = np.eye(4)
    # translation
    transf[0, 3] = xt
    transf[1, 3] = yt
    transf[2, 3] = zt
    # rotation
    rot_x = np.eye(4)
    rot_y = np.eye(4)
    rot_z = np.eye(4)
    if xr != 0:
        rot_x = np.array([[1., 0., 0., 0.], [0., np.cos(xr), -np.sin(xr), 0.], [0., np.sin(xr), np.cos(xr), 0.],
                          [0., 0., 0., 1.]]).astype(float)
    if yr != 0:
        rot_y = np.array([[np.cos(yr), 0., np.sin(yr), 0.], [0., 1., 0., 0.], [-np.sin(yr), 0., np.cos(yr), 0.],
                          [0., 0., 0., 1.]]).astype(float)
    if zr != 0:
        rot_z = np.array([[np.cos(zr), -np.sin(zr), 0., 0.], [np.sin(zr), np.cos(zr), 0., 0.], [0., 0., 1., 0.],
                          [0., 0., 0., 1.]]).astype(float)
    # print transf,rot_x,rot_y,rot_z
    return transf.dot(rot_x).dot(rot_y).dot(rot_z)


def _aff_trans(params, *args):
    transf = aff_tsf(*params)
    coreg_vol = resample_trans(args[1], args[2], args[3], args[4], sw2tw_affine=transf)
    return coreg_vol, transf


def _coreg(params, *args):
    mask_ = args[5]
    coreg_vol, _ = _aff_trans(params, *args)

    #coreg_vol = gaussian_filter(coreg_vol, 0.5, 0)
    #score = np.corrcoef(coreg_vol.ravel(), args[0].ravel())
    score = np.corrcoef(coreg_vol[mask_], args[0][mask_])
    # print score
    # print score[0,1]
    # print x,args
    return 1 - score[0, 1]
    # return score

def rad2deg(params):
    params_c = params.copy()
    params_c[3:, ...] = (params_c[3:, ...] / np.pi) * 180
    return params_c

def fit(source, v2w_source, target, v2w_target, mask = [], verbose = False, stride=2,dowsamp_flag=True):
    # TODO add initialization params for each frame based on the precedent param
    # TODO change size of the target matrix for faster evaluation
    coreg_vols    = []
    transfs       = []
    motion_params = []
    nframes = 1

    if mask==[]:
        mask = np.ones_like(target).astype(bool)
    #else:
    #    mask[::stride, :, :] = False
    #    mask[:, ::stride, :] = False
    #    mask[::stride+1, :, :] = False
    #    mask[:, ::stride+1, :] = False

    if len(source.shape) > 3:
        nframes = source.shape[3]

    if dowsamp_flag:
        # dowsample target
        tv2tw_affine = np.copy(v2w_target)  # np.eye(4)
        tv_shape = np.ceil(np.array(target.shape) / 2.)
        tv2tw_affine[:3, :3] = tv2tw_affine[:3, :3] * 2.

        target_downsamp = resample_trans(target, np.copy(v2w_target), tv2tw_affine, tv_shape)
        #target_downsamp = gaussian_filter(target_downsamp, 0.5, 0)
        #mask_down = np.ones_like(target_downsamp).astype(bool)
        mask_down = resample_trans(mask, np.copy(v2w_target), tv2tw_affine, tv_shape)

    for frame in range(nframes):
        if nframes == 1:
            source_ = source
        else:
            source_ = source[..., frame]


        if dowsamp_flag:
            '''
            # dowsample target
            tv2tw_affine = np.copy(v2w_target)#np.eye(4)
            tv_shape = np.ceil(np.array(target.shape) / 2.)
            tv2tw_affine[:3, :3] = tv2tw_affine[:3, :3] * 2.

            target_downsamp = resample_trans(target, v2w_target, tv2tw_affine, tv_shape)
            #mask = np.ones_like(target_downsamp).astype(bool)
            mask = resample_trans(mask, v2w_target, tv2tw_affine, tv_shape)
            '''
            #source_ = resample_trans(source_, v2w_source, tv2tw_affine, tv_shape)
            #v2w_source = tv2tw_affine
            # Rough estimate
            params = fmin_powell(func=_coreg, x0=np.zeros((1, 6))[0, :],
                                 args=(target_downsamp, source_, v2w_source, tv2tw_affine, target_downsamp.shape, mask_down),
                                 xtol=0.0001, ftol=0.005, disp=verbose)

            # Fine tuned estimate
            #params = fmin_powell(func=_coreg, x0=params,
            #                     args=(target, source_, v2w_source, v2w_target, target.shape, mask),
            #                     xtol=0.0001, ftol=0.005, disp=verbose)
        else:
            params = fmin_powell(func=_coreg, x0=np.zeros((1, 6))[0, :],
                                 args=(target, source_, v2w_source, v2w_target, target.shape, mask),
                                 xtol=0.0001, ftol=0.001, disp=verbose)
        coreg_vol, transf = _aff_trans(params, *(target, source_, v2w_source, v2w_target, target.shape))
        coreg_vols.append(coreg_vol)
        transfs.append(transf)
        motion_params.append(params)
        if nframes == 1:
            motion_params[0] = rad2deg(motion_params[0])
            return coreg_vols[0], transfs[0], motion_params[0]

    motion_params = np.stack(motion_params, axis=1)
    motion_params = rad2deg(motion_params)
    return np.stack(coreg_vols, axis=3), np.stack(transfs, axis=2), motion_params

