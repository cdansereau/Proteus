__author__ = 'Christian Dansereau'
import numpy as np


def getspec(vol):
    nx, ny, nz = vol.shape
    nrows = int(np.ceil(np.sqrt(nz)))
    ncolumns = int(np.ceil(nz / (1. * nrows)))
    return nrows, ncolumns, nx, ny, nz


def montage(vol1):
    vol = np.swapaxes(vol1, 0, 1)
    nrows, ncolumns, nx, ny, nz = getspec(vol)

    mozaic = np.zeros((nrows * nx, ncolumns * ny))
    indx, indy = np.where(np.ones((nrows, ncolumns)))

    for ii in np.arange(vol.shape[2]):
        # we need to flip the image in the x axis
        mozaic[(indx[ii] * nx):((indx[ii] + 1) * nx), (indy[ii] * ny):((indy[ii] + 1) * ny)] = vol[::-1, :, ii]

    return mozaic
