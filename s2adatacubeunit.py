# coding: utf-8
import pick_from_LMSAL
import my_fits
import numpy as np
import matplotlib.pyplot as plt
import alignlib
import rebin
import math
from iris_lmsalpy import extract_irisL2data as ei
from iris_lmsalpy import iris2
from scipy import interpolate, signal
from alignlib import falign, avg_diff
from dateutil import parser
import cv2 as cv
from s2alib import iris2aia, closest_time, sji2aia, template_match
import argparse
import saveblank

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--obsid", required=True,
                help='obsid for input')
args = vars(ap.parse_args())
obsid = args["obsid"]
print("--------HERE---------",obsid)

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

#max_it = max(iris_t_s)
#aia_time2iris = aia_time2iris[0:ei.get_ori_val(aia_time2iris, max_it)+1]

## sji work
sji_path = pick_from_LMSAL.obsid_sji(obsid, pattern='1400')[0]
sji = ei.load(sji_path)
sji1400 = sji.SJI['SJI_1400']
slitx = sji1400.SLTPX1IX
t_sji = sji1400.date_time_acq_ok
sji_yscl = sji1400.SPYSCL
sji_xscl = sji1400.SPCSCL
aux_sji = ei.only_data(sji_path, extension=-2)
sjit_t_s = aux_sji[:,0] #[::-1]
n_rasters = len(pick_from_LMSAL.obsid_raster(obsid, raster=-1))

##iris2aia work
interp0 = []
for i, it in enumerate(t_iris):
	ob = iris2aia(it, iris_xcenix[i], closest_time(it, aia_time2iris), iris_ycenix[i])
#	print(it, aia_time2iris[closest_time(it, aia_time2iris)])
	interp0.append(ob)


interp = []
xcenters = []
ycenters = []

offset = int((sji1400.data.shape[1]-(avg_diff(slitx))*sji1400.data.shape[2])/2) #px
#offset = 230
l_iris =len(t_iris) #count of iris slits
sji_period = l_iris/sji1400.data.shape[2] #how many slits between each sji image = num of filters??

scaled_shape_s2a = (sji1400.data.shape[0]*sji_yscl/aia_yscl, offset*2*sji_xscl/aia_xscl) #sji, shape scaled from sji scale to aia scale 0.6"/px
aux = aia_main[0].copy()

for i in range(len(iris_t_s)):
	closest_sji = closest_time(iris_t_s[i], sjit_t_s)
	# print(i, closest_sji)
	x = slitx[closest_sji]
	nx = round(x)
	s = sji1400.data[:,nx-offset:nx+offset,closest_sji]
	aia_index = closest_time(iris_t_s[i], aia_time2iris)
	aia_index = ei.get_ori_val(aia_time2iris,iris_t_s[i] )
	scaled_img = rebin.congrid(s, scaled_shape_s2a) # slit is not slitx anymore, now it is middle column, scaled to aia scale
	cenx, ceny = template_match(aux, aia_index, scaled_img)
	# print(i, iris_t_s[i], aia_time2iris[aia_index], aia_index, cenx, ceny)
	xcenters.append(cenx)
	ycenters.append(ceny)

newxcenters = xcenters
ycenters = np.array(ycenters)
x = np.arange(0, l_iris, 1)
f = interpolate.interp1d(x, ycenters, fill_value = "extrapolate")
newycenters = f(np.arange(l_iris))

raster = np.zeros((int(scaled_shape_s2a[0]), len(iris_t_s))) # aia scale y, count of iris px x

for i, (xcen, ycen) in enumerate(zip(newxcenters, newycenters)):
	aia_i = interp0[i].aia_time_index
	scy = scaled_shape_s2a[0]/2
#	print(i, aia_i, xcen)
	slit = aia_main[0][aia_i, int(ycen-scy):int(ycen-scy)+int(scaled_shape_s2a[0]), int(xcen):int(xcen)+1].flatten()
	raster[:,i] = slit

plt.ion()
final_s2a_shape = (scaled_shape_s2a[0], abs(iris_map.shape[1]*iris_xscl/aia_xscl)) #final scaled shape in aia scale
final_raster = rebin.congrid(raster, final_s2a_shape)
final_iris = rebin.congrid(iris_map, final_s2a_shape)


#for different channels

channels = ['1700', '304', '193', '171']
paths = [pick_from_LMSAL.obsid_raster2aia(obsid, pattern=p)[0] for p in channels]
mains = [my_fits.read(p, ext=0) for p in paths]
extras = [my_fits.read(p, ext=1) for p in paths]
t2is = [ex[0][:,0] for ex in extras]
rasters = []

for i, channel in enumerate(channels):
	main = mains[i]
	extra = extras[i]
	t2i = t2is[i]
	interp0 = [iris2aia(it, iris_xcenix[i], closest_time(it, t2i), iris_ycenix[i]) for i, it in enumerate(t_iris)]
	araster = np.zeros((int(scaled_shape_s2a[0]), len(newxcenters)))

	for j, (xcen, ycen) in enumerate(zip(newxcenters, newycenters)):
		aia_i = interp0[j].aia_time_index
		scy = scaled_shape_s2a[0]/2
		slit = main[0][aia_i, int(ycen-scy):int(ycen-scy)+int(scaled_shape_s2a[0]), int(xcen):int(xcen)+1].flatten()
		#print(slit)
		araster[:,j] = slit.astype(np.float16)

	rasters.append(rebin.congrid(araster, final_s2a_shape))

ts = [final_iris, final_raster] + rasters
norm = [arr/np.max(arr[:,:])*255 for arr in ts]

# inversion
output = iris2.invert(iris_file)
temp = output.raster['T'].data[:,:,[12,17,22]]
vlos = output.raster['vlos'].data[:,:,[12,17,22]]
vturb = output.raster['vturb'].data[:,:,[12,17,22]]
nne = output.raster['nne'].data[:,:,[12,17,22]]
unc_temp = output.raster['unc_T'].data[:,:,[12,17,22]]
unc_vlos = output.raster['unc_vlos'].data[:,:,[12,17,22]]
unc_vturb = output.raster['unc_vturb'].data[:,:,[12,17,22]]
unc_nne = output.raster['unc_nne'].data[:,:,[12,17,22]]

mask = output.raster['Mg II k 2796'].mask
mask = mask[:,:,np.newaxis]

inv = np.concatenate([temp, vlos, vturb, nne, unc_temp, unc_vlos, unc_vturb, unc_nne, mask], axis=2)
final_inv = []
for i in range(inv.shape[2]):
	final_inv.append(rebin.congrid(inv[:,:,i], final_s2a_shape))

data_cube = norm + final_inv
to_save = np.transpose(np.stack(tuple(data_cube)), [1, 2, 0])
with open('/Users/jkim/Desktop/mg2hk/to_send/data4PChIRIS2_{}.npy'.format(obsid), 'wb') as f:
	np.save(f, to_save)