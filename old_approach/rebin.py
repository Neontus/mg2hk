import scipy.ndimage as ndimage
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Box2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve
from scipy import interpolate


# a=np.arange(24).reshape((4,6))
# rebin.congrid(a, [8,12])
# a=indgen(4,6)
# rebin(a, 12,8)

def make(data_in, factor):
    return ndimage.zoom(data_in, factor)


def rebin(arr, new_shape, mio=False):
    if mio == False: 
        shape = (new_shape[0], arr.shape[0] // new_shape[0] or 1,
                 new_shape[1], arr.shape[1] // new_shape[1] or 1)
        #print(shape)
        return arr.reshape(np.round(shape)).mean(-1).mean(1)
    else:
        print('Mio is done')
        in_shape = arr.shape
        shape = (new_shape[0], np.round(arr.shape[0]//new_shape[0]) or 1,
                 new_shape[1], np.round(arr.shape[1]//new_shape[1]) or 1)
        #print('New... ', shape)
        return arr[0:shape[0]*shape[1]*shape[2]*shape[3]].reshape(np.round(shape)).mean(-1).mean(1)
        #print(arr[0:shape[0]*shape[1]*shape[2]*shape[3]].reshape(np.round(shape)).shape)
        #return np.mean(arr[0:shape[0]*shape[1]*shape[2]*shape[3]].reshape(np.round(shape)), axis=(1, 3))


def congrid(arr, new_shape, method = 'RectBi', kind = 'linear', kx=3, ky=3):
   
    x = np.arange(0, arr.shape[0])
    y = np.arange(0, arr.shape[1])

    if method == 'interp2d': f = interp2d(y, x, arr, kind=kind)
    if method == 'RectBi': f = RectBivariateSpline(x, y, arr, kx=kx, ky=ky)


    #print( arr.shape[0]/new_shape[0], arr.shape[1]/new_shape[1])
    new_x = np.arange(0, arr.shape[0], arr.shape[0]/new_shape[0])
    new_y = np.arange(0, arr.shape[1], arr.shape[1]/new_shape[1])
    #print(new_x.shape, new_y.shape)
    

    return f(new_x,new_y) 


def rebcong(arr, new_shape, factor=10, mio=False, **kwargs):

    dim =  arr.shape
    # shape = ((dim[0]/new_shape[1])*10, (dim[1]/new_shape[1])*10)
    #print(new_shape)
    shape = (np.round(new_shape[0])*factor, np.round(new_shape[1])*factor)
    aux = congrid(arr, shape, **kwargs)
    print(shape)
    return rebin(aux, (shape[0]//factor, shape[1]//factor), mio=mio) #, **kwargs)


def box(arr, nx, ny):

    dim =  arr.shape
    new = np.zeros((dim[0]//ny, dim[1]//nx))
    aa = np.split(arr, np.arange(0,dim[0],ny)[1:])
    print(len(aa))
    count =  0
    bb = []
    coords = []
    for j, i in enumerate(aa):
        cc = np.split(i, np.arange(0,dim[1],nx)[1:], axis=1)
        #for c in cc: bb.append(c)
        for c in cc:
            if c.shape == (ny, nx):
                bb.append(np.nanmean(c))
                new[j, count%(dim[1]//nx)] = np.nanmean(c)
                coords.append([j, count%(dim[1]//nx)])
                count+=1
    return new

def conv_box(arr, n, mode='linear_interp', interpol=False):

    dim =  arr.shape
    nx, ny = n, n
    new_cc = np.zeros((dim[0]//ny, dim[1]//nx))
    px =  Box2DKernel(n, mode=mode)
    out = convolve(arr, px)
    if interpol == True:
        xc = np.arange(out.shape[0])
        yc = np.arange(out.shape[1])
        xn = xc+0.5
        yn = yc+0.5
        xc, yc = np.meshgrid(xc, yc)
        if n % 2 == 0 and interpol == True:
            print('Interpolando...')
            print(xc.shape, yc.shape, out.shape)
            f =interpolate.interp2d(xc, yc, out)
            out = f(xn, yn)
    for ii, i in enumerate(np.arange(0,dim[0],ny)):
        for jj, j in enumerate(np.arange(0,dim[1],nx)):
            py = i+((ny-1)//2)
            px = j+((nx-1)//2)
    #        print(ii,jj,py,px)
            if ii < new_cc.shape[0] and jj < new_cc.shape[1]:
                val = out[py,px]
                if n % 2 == 0:
                   val = np.nanmean(out[py:py+1,px:px+1])
                new_cc[ii,jj] = val
    return new_cc 
