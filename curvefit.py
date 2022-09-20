# IMPORTS
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import cv2

from scipy.optimize import curve_fit
from scipy import ndimage as ndi
from iris_lmsalpy import extract_irisL2data as ei
from aiapy.calibrate import normalize_exposure, register, update_pointing

import rebin
import pick_from_LMSAL
import my_fits

# PARAMS
obsid = '20220626_040436_3620108077'
numraster = 0

iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=numraster)
aia_file = pick_from_LMSAL.obsid_raster2aia(obsid, raster=numraster, pattern='1600')

aia_data = my_fits.read(aia_file[0])
aia_1600 = aia_data[0]
hdr_aia_1600 = aia_data[1]
aia_middle_step = int(aia_1600.shape[0]//2)

aia_1600 = aia_1600[aia_middle_step,:,:]
info_1600 = my_fits.read(aia_file[0], ext=1)
xcen_aia = info_1600[0][:,10]
ycen_aia = info_1600[0][:,11]

hdr_iris_data = ei.only_header(iris_file[0])
aux_hdr_iris_data = ei.only_header(iris_file[0], extension=1)
xcen_iris = hdr_iris_data['XCEN']
ycen_iris = hdr_iris_data['YCEN']
xscl_iris = aux_hdr_iris_data['CDELT3']
yscl_iris = aux_hdr_iris_data['CDELT2']
xscl_aia = hdr_aia_1600['CDELT1']
yscl_aia = hdr_aia_1600['CDELT2']

iris_raster = ei.load(iris_file[0])
extent_hx_hy = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
mgii = iris_raster.raster['Mg II k 2796'].data
sel_wl = ei.get_ori_val(iris_raster.raster['Mg II k 2796'].wl, [2794.73])
# if opt2 == True: sel_wl = ei.get_ori_val(iris_raster.raster['Mg II k 2796'].wl, [2795.99])
limit_mask = np.argmin(np.gradient(np.sum(iris_raster.raster['Mg II k 2796'].mask, axis=1)))
mgii = mgii[:limit_mask,:,sel_wl]
mgii = mgii.clip(75,400)

print("-"*10, "[Section] Scaling IRIS", "-"*10)
new_iris_shape = mgii.shape[0]*yscl_iris, mgii.shape[1]*xscl_iris
new_iris_data = rebin.congrid(mgii, new_iris_shape)

hiris, wiris = new_iris_data.shape
extent_iris = [xcen_iris-wiris/2,xcen_iris+wiris/2,ycen_iris-hiris/2,ycen_iris+hiris/2]
print(extent_iris)
#extent_iris = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
#print(extent_iris)

print("-"*10, "[Section] Scaling AIA", "-"*10)
try_aia = aia_1600.shape[0]*yscl_aia, aia_1600.shape[1]*xscl_aia
new_aia = rebin.congrid(aia_1600, try_aia)
haia, waia = new_aia.shape
extent_aia = [xcen_aia[aia_middle_step]-waia/2,xcen_aia[aia_middle_step]+waia/2,ycen_aia[aia_middle_step]-haia/2,ycen_aia[aia_middle_step]+haia/2]

print("-"*10, "[Section] Cropping AIA to IRIS", "-"*10)
pad = 0
acp = [(extent_iris[0]-pad, extent_iris[3]+pad), (extent_iris[0]-pad, extent_iris[2]-pad), (extent_iris[1]+pad, extent_iris[3]+pad), (extent_iris[1]+pad, extent_iris[2]-pad)]
x_i = int(extent_iris[0]-pad-extent_aia[0])
x_f = int(extent_iris[1]+pad-extent_aia[0])
y_f = -int(extent_iris[3]+pad-extent_aia[3])
y_i = -int(extent_iris[2]-pad-extent_aia[3])
cut_aia = new_aia[y_f:y_i, x_i:x_f]

iris = new_iris_data
aia = cut_aia

def align(aia, scalex = 1, scaley = 1, theta = 0, translatex = 0, translatey = 0):
    flat = np.flatten(aia)

    identity_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    translate_matrix = np.array([
        [1, 0, translatex],
        [0, 1, translatey],
        [0, 0, 1]
    ])

    rotate_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    scale_matrix = np.array([
        [scalex, 0, 0],
        [0, scaley, 0],
        [0, 0, 1]
    ])

    transform = identity_matrix @ translate_matrix @ rotate_matrix @ scale_matrix
    iris = ndi.affine_transform(aia, transform)
    return iris

aligned = align(aia, 1, 1, 0, 0, 5)

print('AIA', aia.shape)
print('IRIS', iris.shape)

fig, ax = plt.subplots(1, 3, figsize=[10, 10])
ax[0].imshow(aia, cmap='afmhot', origin='lower')
ax[1].imshow(iris, cmap='afmhot', origin='lower')
ax[2].imshow(aligned, cmap='afmhot', origin='lower')
plt.show()

# Aligned Images

init_vals = ([1, 1, 0, 0, 5])
limits = ([0.8, 0.8, -5, -10, -10],[1.2, 1.2, 5, 10, 10])

best_values, covar = curve_fit(align, aia, iris, p0=init_vals, bounds=limits)

