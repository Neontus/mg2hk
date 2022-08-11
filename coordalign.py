# PARAMS:

OBSID = '20220626_040436_3620108077'
NUMRASTER = 0
outpath = "/Users/jkim/Desktop/mg2hk/output/"
blur_filter = 15

import os
import numpy as np
import scipy
import matplotlib
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

matplotlib.use('TkAgg')

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

# print(cut_aia)
# print(cut_aia.dtype)

# cv_cut = (cut_aia*8).astype(np.uint8)
# print(cv_cut)
# print(cv_cut.dtype)



# cv_cut_f = cv2.applyColorMap(cv_cut, cv2.COLORMAP_HOT)

# fig, ax = plt.subplots(figsize=[5, 10])
# ax.imshow(cv_cut, cmap='afmhot')


# cv2.imshow("test", cv_cut_f)
# cv2.imshow("test2", new_iris_data)

# plt.show()
# cv2.waitKey(0)

# print(cv_cut.shape)
# print(cv_cut)
# print(cv_cut_f.shape)
# print(cv_cut_f)

# fig, ax = plt.subplots(1, 2, figsize=[10, 10])
# ax[0].imshow(cut_aia, cmap='afmhot', origin='lower')
# ax[1].imshow(new_iris_data, cmap='afmhot', origin='lower')
# plt.show()

# # Decide Contrast for IRIS
# iris_raster.quick_look()
# iris_wl = int(iris_raster.raster['Mg II k 2796'].poi[0].z_pos_ori)
# iris_vmin, iris_vmax = iris_raster.raster['Mg II k 2796'].poi[0].clip_ima

# Deciding Thresholds
print("-"*10, "[Section] Decide Thresholds", "-"*10)
AIA_THRESHOLD, IRIS_THRESHOLDH, IRIS_THRESHOLDL = 0, 100, 0
iris_vmax, iris_vmin = 400, 75
fig, axs = plt.subplots(1, 2, figsize = [10, 10])
aia_im = axs[0].imshow(cut_aia, cmap="afmhot", origin="lower")
iris_im = axs[1].imshow(new_iris_data, cmap="afmhot", origin="lower")

#print(cropped_iris.shape)
#irisbw = cv2.cvtColor(cropped_iris, cv2.COLOR_BGR2GRAY)
plt.subplots_adjust(left=0.25, bottom=0.25)
axaia = plt.axes([0.25, 0.1, 0.65, 0.03])
axiris = plt.axes([0.25, 0.05, 0.65, 0.03])
axblur = plt.axes([0.25, 0.15, 0.65, 0.03])
#axicon = plt.axes([0.25, 0.025, 0.65, 0.03])

aia_slider = plt.Slider(
    ax=axaia,
    label='AIA Threshold',
    valmin=0.1,
    valmax=200,
    valinit=0,
)

iris_slider = RangeSlider(
    ax=axiris,
    label='IRIS thresholds',
    valmin=0.1,
    valmax=600
)

blur_slider = plt.Slider(
    ax=axblur,
    label='Blur Level',
    valmin=0,
    valmax=50,
    valinit=0.1,
)

def update_aia(val):
    global AIA_THRESHOLD
    axs[0].imshow(cut_aia>val, origin="lower", cmap="afmhot")
    AIA_THRESHOLD = val
    fig.canvas.draw_idle()

def update_iris(val):
    global IRIS_THRESHOLDH, IRIS_THRESHOLDL
    IRIS_THRESHOLDL = val[0]
    IRIS_THRESHOLDH = val[1]
    axs[1].imshow(alignlib.imgthr(new_iris_data, IRIS_THRESHOLDL, IRIS_THRESHOLDH), origin="lower", cmap='afmhot')
    #axs[1].imshow(new_iris_data, cmap='afmhot', origin="lower", vmin=val[0], vmax=val[1])
    fig.canvas.draw_idle()
    
def blur_iris(val):
    global blur_filter
    blur_filter = int(val)
    if val<1:
        axs[1].imshow(alignlib.imgthr(new_iris_data, IRIS_THRESHOLDL, IRIS_THRESHOLDH), cmap='afmhot', origin="lower")
        return
        
    axs[1].imshow(alignlib.lee_filter(alignlib.imgthr(new_iris_data, IRIS_THRESHOLDL, IRIS_THRESHOLDH), int(val)), cmap='gray', origin='lower')

    fig.canvas.draw_idle()
    
aia_slider.on_changed(update_aia)
iris_slider.on_changed(update_iris)
blur_slider.on_changed(blur_iris)

plt.show()


# Prepare Images for Alignment
print("-"*10, "[Section] Prepare Images for Alignment", "-"*10)
aia_to_align = ((cut_aia>AIA_THRESHOLD)*255).astype(np.uint8)
iris_to_align = cv2.normalize(alignlib.lee_filter((alignlib.imgthr(new_iris_data, IRIS_THRESHOLDL, IRIS_THRESHOLDH)*255), blur_filter), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# Running Alignment
print("-"*10, "[Section] Aligning Images", "-"*10)
matrix, walign, halign = alignlib.align(aia_to_align, iris_to_align, debug=True, num_max_points = 10, blurFilter = blur_filter)

aligned_color_aia = cv2.warpAffine(cut_aia, matrix, (walign, halign))
aligned_aia = cv2.warpAffine(aia_to_align, matrix, (walign, halign))

# Results
print("-"*10, "[Section] Results", "-"*10)
print("ALIGNED AIA ARRAY SIZE: ", aligned_color_aia.shape)
print("IRIS ARRAY SIZE: ", new_iris_data.shape)
print("Transformation Matrix: ", matrix)

# fig, ax = plt.subplots(1, 5, figsize=[5, 10])
# ax[0].imshow(aligned_color_aia, cmap="afmhot", origin="lower")
# ax[1].imshow(aligned_aia, cmap="afmhot", origin="lower")
# ax[2].imshow(new_iris_data, cmap='afmhot', origin="lower", vmin = iris_vmin, vmax = iris_vmax)
# ax[3].imshow(color_aia_to_align, cmap="afmhot", origin="lower")
# ax[4].imshow(cut_aia, cmap="afmhot", origin="lower")
# plt.show()


# Flickering
print("-"*10, "[Section] Flickering", "-"*10)

fig, ax = plt.subplots(1, 1, figsize=[5, 10])
toggle=0


while True:
    if (toggle%2 == 0):
        ax.imshow(aligned_color_aia, cmap="afmhot", origin="lower")
    else:
        ax.imshow(new_iris_data, cmap='afmhot', origin="lower", vmin = iris_vmin, vmax = iris_vmax)
    display.display(fig)
    display.clear_output(wait = True)
    toggle += 1
    plt.pause(0.5)
    
# # Getting image arrays for comparison
# print("-"*10, "[Section] Evaluating Similarity", "-"*10)
# # cropped_iris, reshaped_result