import numpy as np
from iris_lmsalpy import extract_irisL2data as ei
import rebin
import pick_from_LMSAL
import my_fits
import alignlib
import cv2
import matplotlib.pyplot as plt
from IPython import display

import argparse
IRIS_THRESHOLDH = 450.025

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--obsid", required=True,
                help='obsid for input')
ap.add_argument("-a", "--aia", required=True,
                help='aia for input')
ap.add_argument("-i", "--iris", required=True,
                help='iris for input')
ap.add_argument("-b", "--blur", required=True,
                help='blur for input')


args = vars(ap.parse_args())

OBSID = args["obsid"]
init_aia_n = args["aia"]
init_iris_n = args["iris"]
init_blur = args["blur"]

print("testing with: (OBSID - {})".format(OBSID))

NUMRASTER = 0

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


a = alignlib.super_align(cut_aia, new_iris_data, init_aia_n, init_iris_n, init_blur)
res = a.minimize()
best_params = res["x"]

best_aia_N, best_iris_N, best_blur = best_params

AIA_THRESH = alignlib.get_top_n(cut_aia, best_aia_N)
IRIS_THRESH_L = alignlib.get_top_n(new_iris_data, best_iris_N)
IRIS_THRESH_H = 450.025

aia_to_align = ((cut_aia > AIA_THRESH) * 255).astype(np.uint8)
iris_to_align = cv2.normalize(alignlib.lee_filter((alignlib.imgthr(new_iris_data, IRIS_THRESH_L, IRIS_THRESH_H) * 255), best_blur), None, 0,255, cv2.NORM_MINMAX).astype('uint8')

matrix, walign, halign = alignlib.sift_ransac(aia_to_align, iris_to_align, debug=False)

aligned_color_aia = cv2.warpAffine(cut_aia, matrix, (walign, halign))

fig, ax = plt.subplots(1, 1, figsize=[5, 10])
toggle = 0
iris_vmax, iris_vmin = 400, 75


for i in range(10):
    if (toggle % 2 == 0):
        ax.imshow(aligned_color_aia, cmap="afmhot", origin="lower")
    else:
        ax.imshow(new_iris_data, cmap='afmhot', origin="lower", vmin=iris_vmin, vmax=iris_vmax)
    display.display(fig)
    display.clear_output(wait=True)
    toggle += 1
    plt.pause(0.5)

