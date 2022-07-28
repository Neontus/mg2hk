# PARAMS:

OBSID = '20220709_210913_3620108077'
NUMRASTER = 0
outpath = "/Users/jkim/Desktop/mg2hk/output/"

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
from matplotlib.widgets import RangeSlider

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

# Shaping AIA
print("-"*10, "[Section] Shaping AIA", "-"*10)
try_aia = aia_1600.shape[0]*yscl_aia, aia_1600.shape[1]*xscl_aia
new_aia = rebin.congrid(aia_1600, try_aia)
haia, waia = new_aia.shape
extent_aia = [xcen_aia[aia_middle_step]-waia/2,xcen_aia[aia_middle_step]+waia/2,ycen_aia[aia_middle_step]-haia/2,ycen_aia[aia_middle_step]+haia/2]

# Shaping IRIS
print("-"*10, "[Section] Shaping IRIS", "-"*10)
iris_raster = ei.load(iris_file[0])
extent_hx_hy = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
iris_data = iris_raster.raster['Mg II k 2796'].data

# Decide Frequency
iris_raster.quick_look()
iris_freq = int(iris_raster.raster['Mg II k 2796'].poi[0].z_pos_ori)
iris_vmin, iris_vmax = iris_raster.raster['Mg II k 2796'].poi[0].clip_ima

new_iris_shape = iris_data[:,:,iris_freq].shape[0]*yscl_iris, iris_data[:,:,iris_freq].shape[1]*xscl_iris
new_iris_data = rebin.congrid(iris_data[:,:,iris_freq], new_iris_shape)

# Cropping IRIS
print("-"*10, "[Checkpoint] Cropping IRIS", "-"*10)
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

cropped_iris = crop_image(new_iris_data)
hiris, wiris = cropped_iris.shape
extent_iris = [xcen_iris-wiris/2,xcen_iris+wiris/2,ycen_iris-hiris/2,ycen_iris+hiris/2]

# Cutting AIA to IRIS
print("-"*10, "[Section] Cutting AIA to IRIS shape", "-"*10)
acp = [(extent_iris[0]-5, extent_iris[3]+5), (extent_iris[0]-5, extent_iris[2]-5), (extent_iris[1]+5, extent_iris[3]+5), (extent_iris[1]+5, extent_iris[2]-5)]
x_i = int(extent_iris[0]-5-extent_aia[0])
x_f = int(extent_iris[1]+5-extent_aia[0])
y_f = -int(extent_iris[3]+5-extent_aia[3])
y_i = -int(extent_iris[2]-5-extent_aia[3])

cut_aia = new_aia[y_f:y_i, x_i:x_f]

# Deciding Thresholds
print("-"*10, "[Section] Decide Thresholds", "-"*10)
AIA_THRESHOLD, IRIS_THRESHOLD = 0, 0
fig, axs = plt.subplots(1, 2, figsize = [10, 10])
aia_im = axs[0].imshow(cut_aia, cmap="afmhot", origin="lower")
iris_im = axs[1].imshow(cropped_iris, cmap="afmhot", origin="lower")
plt.subplots_adjust(left=0.25, bottom=0.25)
axaia = plt.axes([0.25, 0.1, 0.65, 0.03])
axiris = plt.axes([0.25, 0.05, 0.65, 0.03])

aia_slider = plt.Slider(
    ax=axaia,
    label='AIA Threshold',
    valmin=0.1,
    valmax=200,
    valinit=0,
)

iris_slider = plt.Slider(
    ax=axiris,
    label='IRIS Threshold',
    valmin=0.1,
    valmax=200,
    valinit=0,
)

def update_aia(val):
    global AIA_THRESHOLD
    axs[0].imshow(cut_aia>val, cmap="afmhot", origin="lower")
    AIA_THRESHOLD = val
    fig.canvas.draw_idle()

def update_iris(val):
    global IRIS_THRESHOLD
    axs[1].imshow(cropped_iris>val, cmap="afmhot", origin="lower")
    IRIS_THRESHOLD = val
    fig.canvas.draw_idle()
    
aia_slider.on_changed(update_aia)
iris_slider.on_changed(update_iris)
plt.show()
plt.close()


# Prepare Images for Alignment
print("-"*10, "[Section] Prepare Images for Alignment", "-"*10)
# aia_thres = cut_aia>70
# iris_thres = cropped_iris>12


fig, ax = plt.subplots(figsize=[10,10])
ax.imshow(cut_aia, cmap='afmhot', origin="lower")
sb.saveblank(outpath, "aia_color_gen_coord_"+OBSID)

fig, ax = plt.subplots(figsize=[10,10])
ax.imshow(cut_aia>AIA_THRESHOLD, cmap='afmhot', origin="lower")
sb.saveblank(outpath, "aia_gen_coord_"+OBSID)

fig, ax = plt.subplots(figsize=[10,10])
ax.imshow(cropped_iris>IRIS_THRESHOLD, origin="lower", cmap="afmhot")
sb.saveblank(outpath, "iris_gen_coord_"+OBSID)

color_aia_to_align = cv2.imread(outpath + "aia_color_gen_coord_{}.png".format(OBSID))
aia_to_align = cv2.imread(outpath + "aia_gen_coord_{}.png".format(OBSID))
iris_to_align = cv2.imread(outpath + "iris_gen_coord_{}.png".format(OBSID))

# Running Alignment
print("-"*10, "[Section] Aligning Images", "-"*10)
alignlib.align_images(color_aia_to_align, aia_to_align, iris_to_align,"/Users/jkim/Desktop/mg2hk/output/genresult_{}.png".format(OBSID), 150, True, 50)

# Flickering
print("-"*10, "[Section] Flickering", "-"*10)
outputpath = "/Users/jkim/Desktop/mg2hk/output/genresult_{}.png".format(OBSID)
result = mpimg.imread(outputpath)
new_result = result[:,:,0]
newest = crop_image(new_result)
reshaped_result = rebin.congrid(newest, cropped_iris.shape)
fig, ax = plt.subplots(1, 1, figsize=[5, 10])

for i in range(10):
    if (i%2 == 0):
        ax.imshow(reshaped_result, cmap="afmhot")
    else:
        ax.imshow(cropped_iris, cmap='afmhot', origin="lower", vmin = iris_vmin, vmax = iris_vmax)
    display.display(fig)
    display.clear_output(wait = True)
    plt.pause(0.5)

# Getting image arrays for comparison
print("-"*10, "[Section] Evaluating Similarity", "-"*10)
# cropped_iris, reshaped_result