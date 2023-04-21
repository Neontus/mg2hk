# coding: utf-8
import pick_from_LMSAL
import my_fits
import numpy as np
import matplotlib.pyplot as plt
import alignlib
import rebin
import math
from iris_lmsalpy import extract_irisL2data as ei
from scipy import interpolate, signal
from alignlib import falign, avg_diff
from dateutil import parser
import cv2 as cv


obsid = '20230103_194208_3610108077'

path = pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')[0]
aia_main = my_fits.read(path, ext=0)
aia_extra = my_fits.read(path, ext=1)
aia_time2iris = aia_extra[0][:,0]
aia_xscl = aia_main[1]['CDELT1']
aia_yscl = aia_main[1]['CDELT2']

## iris work
iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=0)[0]
iris_raster = ei.load(iris_file)
# hiris, wiris = iris_raster.raster['Mg II k 2796']['extent_heliox_helioy'][3]-iris_raster.raster['Mg II k 2796']['extent_heliox_helioy'][2], iris_raster.raster['Mg II k 2796']['extent_heliox_helioy'][1]-iris_raster.raster['Mg II k 2796']['extent_heliox_helioy'][0]
# start_iris2aia = np.argmin(np.abs(aia_extra[0][:,0]))
iris_t_s = iris_raster.raster['Mg II k 2796'].date_time_acq_in_seconds
iris_xcenix = iris_raster.raster['Mg II k 2796'].XCENIX
iris_ycenix = iris_raster.raster['Mg II k 2796'].YCENIX
iris_xscl = iris_raster.raster['Mg II k 2796'].SPXSCL
iris_yscl = iris_raster.raster['Mg II k 2796'].SPYSCL
t_iris = iris_raster.raster['Mg II k 2796']['date_time_acq_in_seconds']
wl_iris = iris_raster.raster['Mg II k 2796'].wl
pos = ei.get_ori_val(wl_iris, 2794.)
iris_map = iris_raster.raster['Mg II k 2796'].data[:,:,pos]
extent_iris_arcsec = iris_raster.raster['Mg II k 2796'].extent_arcsec_arcsec


## sji work
sji_path = pick_from_LMSAL.obsid_sji(obsid, pattern='1400')[0]
sji = ei.load(sji_path)
sji1400 = sji.SJI['SJI_1400']
slitx = sji1400.SLTPX1IX
t_sji = sji1400.date_time_acq
sji_yscl = sji1400.SPYSCL
sji_xscl = sji1400.SPCSCL

##iris2aia work
interp0 = []
class iris2aia:
	def __init__(self, t_iris, x_iris, i_aia, y_iris = None):
		self.iris_time = t_iris
		self.iris_x = x_iris
		self.aia_time_index = i_aia
		self.iris_y = y_iris

	def add_iris_x(self, new_iris_x):
		self.new_iris_x = new_iris_x

def closest_time(iris_time, aia_times):
	differences = [abs(iris_time - aia_time) for aia_time in aia_times]
	# print("time difference: ", min(differences))
	return differences.index(min(differences))

for i, it in enumerate(t_iris):
	ob = iris2aia(it, iris_xcenix[i], closest_time(it, aia_time2iris), iris_ycenix[i])
	interp0.append(ob)


interp = []

class sji2aia:
	def __init__(self, sji_img, iris_t, i_aia):
		self.sji_img = sji_img
		self.iris_t = iris_t
		self.aia_i = i_aia

def template_match(aia_index, sji_img):
	method = eval('cv.TM_CCOEFF_NORMED')
	aia_img = aia_main[0][aia_index, :,:].astype(np.uint8)
	res = cv.matchTemplate(sji_img.astype(np.uint8), aia_img, method)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

	if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
		top_left = min_loc
	else:
		top_left = max_loc

	#bottom_right = (top_left[0] + sji_img.shape[1], top_left[1] + sji_img.shape[0])
	# cv.rectangle(aia_img, top_left, bottom_right, (255,0,0), 2)
	# plt.subplot(131), plt.imshow(res, cmap='gray')
	# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	# plt.subplot(132), plt.imshow(aia_img.astype(np.uint8), cmap='gray')
	# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	# plt.subplot(133), plt.imshow(sji_img, cmap='gray')
	# plt.show()

	cx, cy = top_left[0] + sji_img.shape[1]/2, top_left[1]+sji_img.shape[0]/2

	return cx, cy

xcenters = []
ycenters = []

offset = int((sji1400.data.shape[1]-(avg_diff(slitx))*sji1400.data.shape[2])/2)
#offset = 230
l_iris =len(t_iris)
sji_period = l_iris/sji1400.data.shape[2]

scaled_shape_s2a = (sji1400.data.shape[0]*sji_yscl/aia_yscl, offset*2*sji_xscl/aia_xscl)

for i, x in enumerate(slitx):
    nx = round(x)
    s = sji1400.data[:,nx-offset:nx+offset,i]
    sjit = iris_t_s[int((i+1)*(len(iris_t_s)/len(slitx))-1)] ##innaccuracy??
    aia_index = closest_time(sjit, aia_time2iris)
    interp.append(sji2aia(s, t_sji[i], aia_index))

    scaled_img = rebin.congrid(s, scaled_shape_s2a) # slit is not slitx anymore, now it is middle column
    cenx, ceny = template_match(aia_index, scaled_img)
    xcenters.append(cenx)
    ycenters.append(ceny)

xcenters = np.array(xcenters)
x = np.arange(0, l_iris, sji_period)
f = interpolate.interp1d(x, xcenters, fill_value = "extrapolate")
newxcenters = f(np.arange(l_iris))

ycenters = np.array(ycenters)
x = np.arange(0, l_iris, sji_period)
f = interpolate.interp1d(x, ycenters, fill_value = "extrapolate")
newycenters = f(np.arange(l_iris))

raster = np.zeros((int(scaled_shape_s2a[0]), len(newxcenters)))

for i, (xcen, ycen) in enumerate(zip(newxcenters, newycenters)):
	aia_i = interp0[i].aia_time_index
	scy = scaled_shape_s2a[0]/2
	slit = aia_main[0][aia_i, int(ycen-scy):int(ycen-scy)+int(scaled_shape_s2a[0]), int(xcen):int(xcen)+1].flatten()
	raster[:,i] = slit

plt.ion()
fig, ax = plt.subplots(1,2,figsize=[10,8], sharey=True, sharex=True)
fig.suptitle('OBSID: '+obsid)
#extent_sji_arcsec = [0,,0,len(newxcenters)]
#con_iris_map = rebin.congrid(iris_map, (h, w))

#scale iris array to same as aia using scales
new_iris_shape = (iris_map.shape[0]*iris_yscl/aia_yscl, len(newxcenters))
iris_to_align = rebin.congrid(iris_map, new_iris_shape)

ax[0].imshow(raster, origin='lower'); ax[0].set_title('synthetic sji-aligned aia raster')
ax[1].imshow(iris_to_align, vmin=90, vmax=300, origin='lower'); ax[1].set_title('aiapx-scaled iris map')

# # masking iris map
# # plt.imshow(np.ma.masked_where(np.invert(mask), iris_map), vmin=90, vmax=300, origin='lower', extent=extent_iris_arcsec)
# mask = iris_raster.raster['Mg II k 2796'].mask
# masked_iris_map = np.ma.masked_where(np.invert(mask), iris_map)
# fin_iris_map = masked_iris_map[~np.isnan(masked_iris_map).all(axis=1)]


