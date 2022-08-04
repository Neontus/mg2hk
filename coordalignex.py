# PARAMS:

OBSID = '20220626_040436_3620108077'
NUMRASTER = 0
outpath = "/Users/jkim/Desktop/mg2hk/output/"
blur_filter = 5

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import cv2
import warnings

from iris_lmsalpy import extract_irisL2data as ei
from aiapy.calibrate import normalize_exposure, register, update_pointing
from astropy.io import fits
from IPython import display
from matplotlib.widgets import RangeSlider, Button

import rebin
import pick_from_LMSAL
import my_fits
import saveblank as sb
import alignlib


print("-"*10, "[Section] Loading Data", "-"*10)
#obsid, numraster = '20211015_051453_3600009176', 0
obsid, numraster = OBSID, NUMRASTER
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

# Mask + Crop IRIS
print("-"*10, "[Section] IRIS Masking + Cropping + WL", "-"*10)
iris_raster = ei.load(iris_file[0])
extent_hx_hy = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
mgii = iris_raster.raster['Mg II k 2796'].data
sel_wl = ei.get_ori_val(iris_raster.raster['Mg II k 2796'].wl, [2794.73])
limit_mask = np.argmin(np.gradient(np.sum(iris_raster.raster['Mg II k 2796'].mask, axis=1)))
mgii = mgii[:limit_mask,:,sel_wl]
mgii = mgii.clip(75,400)

# Scale IRIS to arcsec
new_iris_shape = mgii.shape[0]*yscl_iris, mgii.shape[1]*xscl_iris
new_iris_data = rebin.congrid(mgii, new_iris_shape)

hiris, wiris = new_iris_data.shape
extent_iris = [xcen_iris-wiris/2,xcen_iris+wiris/2,ycen_iris-hiris/2,ycen_iris+hiris/2]
print(extent_iris)
extent_iris = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
print(extent_iris)

# Scale AIA to arcsec
print("-"*10, "[Section] Reshaping AIA to IRIS", "-"*10)
try_aia = aia_1600.shape[0]*yscl_aia, aia_1600.shape[1]*xscl_aia
new_aia = rebin.congrid(aia_1600, try_aia)
haia, waia = new_aia.shape
print('AIA size', haia, waia)
extent_aia = [xcen_aia[aia_middle_step]-waia/2,xcen_aia[aia_middle_step]+waia/2,ycen_aia[aia_middle_step]-haia/2,ycen_aia[aia_middle_step]+haia/2]

# Cropping AIA to IRIS
print("-"*10, "[Section] Cropping AIA to IRIS", "-"*10)
pad = 10
acp = [(extent_iris[0]-pad, extent_iris[3]+pad), (extent_iris[0]-pad, extent_iris[2]-pad), (extent_iris[1]+pad, extent_iris[3]+pad), (extent_iris[1]+pad, extent_iris[2]-pad)]
x_i = int(extent_iris[0]-pad-extent_aia[0])
x_f = int(extent_iris[1]+pad-extent_aia[0])
y_f = -int(extent_iris[3]+pad-extent_aia[3])
y_i = -int(extent_iris[2]-pad-extent_aia[3])
cut_aia = new_aia[y_f:y_i, x_i:x_f]
print('Size cutout AIA', cut_aia.shape)
fig, ax = plt.subplots(1, 2, figsize=[10, 10])
ax[0].imshow(cut_aia, cmap='afmhot', origin='lower')
ax[1].imshow(new_iris_data, cmap='afmhot', origin='lower')
plt.show()

# Prepare Images for Alignment
print("-"*10, "[Section] Prepare Images for Alignment", "-"*10)
fig, ax = plt.subplots(figsize=[10,10])
ax.imshow(cut_aia, cmap='afmhot', origin="lower")
sb.saveblank(outpath, "ex_aia_color_gen_coord_"+OBSID)

fig, ax = plt.subplots(figsize=[10,10])
ax.imshow(cut_aia, cmap='afmhot', origin="lower")
sb.saveblank(outpath, "ex_aia_gen_coord_"+OBSID)

fig, ax = plt.subplots(figsize=[10,10])
ax.imshow(new_iris_data, origin="lower", cmap="afmhot")
sb.saveblank(outpath, "ex_iris_gen_coord_"+OBSID)

color_aia_to_align = cv2.imread(outpath + "ex_aia_color_gen_coord_{}.png".format(OBSID))
aia_to_align = cv2.imread(outpath + "ex_aia_gen_coord_{}.png".format(OBSID))
iris_to_align = cv2.imread(outpath + "ex_iris_gen_coord_{}.png".format(OBSID))

# Running Alignment
print("-"*10, "[Section] Aligning Images", "-"*10)
alignlib.align_images(color_aia_to_align, aia_to_align, iris_to_align,"/Users/jkim/Desktop/mg2hk/output/ex_genresult_{}.png".format(OBSID), 150, True, 70, blur_filter)

# Flickering
def crop_image(img,tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

print("-"*10, "[Section] Flickering", "-"*10)
outputpath = "/Users/jkim/Desktop/mg2hk/output/ex_genresult_{}.png".format(OBSID)
result = mpimg.imread(outputpath)
new_result = result[:,:,0]
newest = crop_image(new_result)
reshaped_result = rebin.congrid(newest, new_iris_data.shape)
fig, ax = plt.subplots(1, 1, figsize=[5, 10])

toggle=0

try:
    while True:
        if (toggle%2 == 0):
            ax.imshow(reshaped_result, cmap="afmhot")
        else:
            ax.imshow(new_iris_data, cmap='afmhot', origin="lower")
        display.display(fig)
        display.clear_output(wait = True)
        toggle += 1
        plt.pause(0.5)
except KeyboardInterrupt:
    pass    
    

# Getting image arrays for comparison
print("-"*10, "[Section] Evaluating Similarity", "-"*10)
# cropped_iris, reshaped_result