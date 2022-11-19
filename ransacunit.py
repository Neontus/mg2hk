#printing

import sys, os

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()


# PARAMS:
import argparse
IRIS_THRESHOLDH = 450.025

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--obsid", required=True,
                help='obsid for input')
ap.add_argument("-b", "--blur", required=True,
                help="blur filter size")
ap.add_argument("-n", "--aian", required=True,
                help="top n for pixels")
ap.add_argument("-r", "--irisn", required=True,
                help="top n for pixels(iris)")
args = vars(ap.parse_args())

OBSID = args["obsid"]
aia_N = float(args["aian"])
iris_N = float(args["irisn"])
blur_filter = int(args["blur"])

print("testing with: (OBSID - {}), (N - {}), (blur - {})".format(OBSID, [aia_N, iris_N], blur_filter))
#
NUMRASTER = 0
outpath = "/Users/jkim/Desktop/mg2hk/output/"

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
import alignlib

# matplotlib.use("TkAgg")

print("-" * 10, "[Section] Loading Data", "-" * 10)
# obsid, numraster = '20211015_051453_3600009176', 0
obsid, numraster = OBSID, NUMRASTER
iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=numraster)
aia_file = pick_from_LMSAL.obsid_raster2aia(obsid, raster=numraster, pattern='1600')

aia_data = my_fits.read(aia_file[0])
aia_1600 = aia_data[0]
hdr_aia_1600 = aia_data[1]
aia_middle_step = int(aia_1600.shape[0] // 2)

aia_1600 = aia_1600[aia_middle_step, :, :]
info_1600 = my_fits.read(aia_file[0], ext=1)
xcen_aia = info_1600[0][:, 10]
ycen_aia = info_1600[0][:, 11]

hdr_iris_data = ei.only_header(iris_file[0])
aux_hdr_iris_data = ei.only_header(iris_file[0], extension=1)
xcen_iris = hdr_iris_data['XCEN']
ycen_iris = hdr_iris_data['YCEN']
xscl_iris = aux_hdr_iris_data['CDELT3']
yscl_iris = aux_hdr_iris_data['CDELT2']
xscl_aia = hdr_aia_1600['CDELT1']
yscl_aia = hdr_aia_1600['CDELT2']

# Mask + Crop IRIS
print("-" * 10, "[Section] IRIS Masking + Cropping + WL", "-" * 10)
iris_raster = ei.load(iris_file[0])
extent_hx_hy = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
mgii = iris_raster.raster['Mg II k 2796'].data
sel_wl = ei.get_ori_val(iris_raster.raster['Mg II k 2796'].wl, [2794.73])
# if opt2 == True: sel_wl = ei.get_ori_val(iris_raster.raster['Mg II k 2796'].wl, [2795.99])
limit_mask = np.argmin(np.gradient(np.sum(iris_raster.raster['Mg II k 2796'].mask, axis=1)))
mgii = mgii[:limit_mask, :, sel_wl]
mgii = mgii.clip(75, 400)

# Scale IRIS to arcsec
new_iris_shape = mgii.shape[0] * yscl_iris, mgii.shape[1] * xscl_iris
new_iris_data = rebin.congrid(mgii, new_iris_shape)

hiris, wiris = new_iris_data.shape
extent_iris = [xcen_iris - wiris / 2, xcen_iris + wiris / 2, ycen_iris - hiris / 2, ycen_iris + hiris / 2]
print(extent_iris)
# extent_iris = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
# print(extent_iris)

# Scale AIA to arcsec
print("-" * 10, "[Section] Reshaping AIA to IRIS", "-" * 10)
try_aia = aia_1600.shape[0] * yscl_aia, aia_1600.shape[1] * xscl_aia
new_aia = rebin.congrid(aia_1600, try_aia)
haia, waia = new_aia.shape
print('AIA size', haia, waia)
extent_aia = [xcen_aia[aia_middle_step] - waia / 2, xcen_aia[aia_middle_step] + waia / 2,
              ycen_aia[aia_middle_step] - haia / 2, ycen_aia[aia_middle_step] + haia / 2]

# Cropping AIA to IRIS
print("-" * 10, "[Section] Cropping AIA to IRIS", "-" * 10)
pad = 0
acp = [(extent_iris[0] - pad, extent_iris[3] + pad), (extent_iris[0] - pad, extent_iris[2] - pad),
       (extent_iris[1] + pad, extent_iris[3] + pad), (extent_iris[1] + pad, extent_iris[2] - pad)]
x_i = int(extent_iris[0] - pad - extent_aia[0])
x_f = int(extent_iris[1] + pad - extent_aia[0])
y_f = -int(extent_iris[3] + pad - extent_aia[3])
y_i = -int(extent_iris[2] - pad - extent_aia[3])
cut_aia = new_aia[y_f:y_i, x_i:x_f]

print('Size cutout AIA', cut_aia.shape)

# Prepare Images for Alignment
print("-" * 10, "[Section] Prepare Images for Alignment", "-" * 10)

AIA_THRESHOLD = alignlib.get_top_n(cut_aia, aia_N)
IRIS_THRESHOLDL = alignlib.get_top_n(new_iris_data, iris_N)

aia_to_align = ((cut_aia > AIA_THRESHOLD) * 255).astype(np.uint8)
iris_to_align = cv2.normalize(
    alignlib.lee_filter((alignlib.imgthr(new_iris_data, IRIS_THRESHOLDL, IRIS_THRESHOLDH) * 255), blur_filter), None, 0,
    255, cv2.NORM_MINMAX).astype('uint8')

# Testing Defaults
iris_vmax, iris_vmin = 400, 75
print("""AIA THRESHOLD: {}
BLUR: {}
IRIS_THRESHOLD: {}""".format(AIA_THRESHOLD, blur_filter, (IRIS_THRESHOLDL, IRIS_THRESHOLDH)))

# Running Alignment
print("-" * 10, "[Section] Aligning Images", "-" * 10)
# matrix, walign, halign = alignlib.align(aia_to_align, iris_to_align, debug=True, num_max_points = 5, blurFilter = blur_filter)

matrix, walign, halign = alignlib.sift_ransac(aia_to_align, iris_to_align, debug=False)

aligned_color_aia = cv2.warpAffine(cut_aia, matrix, (walign, halign))
aligned_aia = cv2.warpAffine(aia_to_align, matrix, (walign, halign))

# Results
print("-" * 10, "[Section] Results", "-" * 10)
print("ALIGNED AIA ARRAY SIZE: ", aligned_color_aia.shape)
print("IRIS ARRAY SIZE: ", new_iris_data.shape)
print("Transformation Matrix: ", matrix)
print("""AIA THRESHOLD: {}
BLUR FILTER: {}
IRIS THRESHOLDS: {}""".format(AIA_THRESHOLD, blur_filter, (IRIS_THRESHOLDL, IRIS_THRESHOLDH)))
error = alignlib.mse(aligned_color_aia, new_iris_data)

enablePrint()

print("N: ", aia_N, iris_N)
print("Blur: ", blur_filter)
print("MSE: ", error)

blockPrint()

# Flickering
# print("-" * 10, "[Section] Flickering", "-" * 10)

fig, ax = plt.subplots(1, 1, figsize=[5, 10])
toggle = 0

for i in range(10):
    if (toggle % 2 == 0):
        ax.imshow(aligned_color_aia, cmap="afmhot", origin="lower")
    else:
        ax.imshow(new_iris_data, cmap='afmhot', origin="lower", vmin=iris_vmin, vmax=iris_vmax)
    display.display(fig)
    display.clear_output(wait=True)
    toggle += 1
    plt.pause(0.5)

