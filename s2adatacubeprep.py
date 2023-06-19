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

# obsid = '20230415_121039_3400109477'
# obsid = '20230515_193520_3620112077'
# obsid = '20220111_012752_3620108076' ehhh
# obsid = '20230509_042004_3882010194' low res??
# obsid = '20230508_182225_3660109102' error
# obsid = '20230411_211053_3400107460' error
# obsid = '20230412_080000_3600100038' error/low res
# obsid = '20230415_082731_3400509477' weird
# obsid = '20230326_040549_3882010194' error
# obsid = '20230327_001440_3660259103' error
# obsid = '20230329_155201_3400113362'
# obsid = '20230404_030531_3400109477'
# obsid = '20221205_211856_3620104077'
# obsid = '20200402_060504_3610108077'
obsid = '20230218_012527_3620108076'

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
#aux = aux[::-1,:,:]
#aux_sji = sji1400.data[:,:,::-1]

for i in range(len(iris_t_s)):
	closest_sji = closest_time(iris_t_s[i], sjit_t_s)
	# print(i, closest_sji)
	x = slitx[closest_sji]
	nx = round(x)
	s = sji1400.data[:,nx-offset:nx+offset,closest_sji]
	#s = sji1400.data[:,nx-offset:nx+offset,int(i/2.)]
	#s = aux_sji[:,nx-offset:nx+offset,i]
	#sjit = iris_t_s[int((i+1)*(len(iris_t_s)/len(slitx))-1)] ##innaccuracy??
	#print(closest_sji)
	#aia_index = closest_time(sjit_t_s[closest_sji], aia_time2iris)
	aia_index = closest_time(iris_t_s[i], aia_time2iris)
	aia_index = ei.get_ori_val(aia_time2iris,iris_t_s[i] )
	#print(i, iris_t_s[i], aia_time2iris[aia_index])
	#interp.append(sji2aia(s, sjit_t_s[closest_sji], aia_index))
	scaled_img = rebin.congrid(s, scaled_shape_s2a) # slit is not slitx anymore, now it is middle column, scaled to aia scale
	#cenx, ceny = template_match(aia_main, aia_index, scaled_img)
	cenx, ceny = template_match(aux, aia_index, scaled_img)
	# print(i, iris_t_s[i], aia_time2iris[aia_index], aia_index, cenx, ceny)
	xcenters.append(cenx)
	ycenters.append(ceny)

# xcenters = np.array(xcenters)
# x = np.arange(0, l_iris, sji_period)
# f = interpolate.interp1d(x, xcenters, fill_value = "extrapolate")
# newxcenters = f(np.arange(l_iris))

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
# fig, ax = plt.subplots(1,2,figsize=[10,8], sharey=True, sharex=True)
# fig.suptitle('OBSID: '+obsid)
# #extent_sji_arcsec = [0,,0,len(newxcenters)]
# #con_iris_map = rebin.congrid(iris_map, (h, w))

# #scale iris array to same as aia using scales
# new_iris_shape = (iris_map.shape[0]*iris_yscl/aia_yscl, len(newxcenters))
# iris_to_align = rebin.congrid(iris_map, new_iris_shape)

# ax[0].imshow(raster, origin='lower'); ax[0].set_title('synthetic sji-aligned aia raster')
# ax[1].imshow(iris_to_align, vmin=90, vmax=300, origin='lower'); ax[1].set_title('aiapx-scaled iris map')

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
# norm = [arr/np.max(arr[:,:])*255 for arr in ts]

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

data_cube = ts + final_inv

m = rebin.congrid(mask, final_s2a_shape)>0.1
w = np.where(m != 0)
masked = [aux[w[0][0]:w[0][-1], w[1][0]:w[1][-1]] for aux in data_cube]

normalizing_factor = [np.nanmax(x[:,:]) for x in masked]
normalized = [arr/normalizing_factor[i]*255 for i,arr in enumerate(masked)]

to_save = np.transpose(np.stack(tuple(normalized)), [1, 2, 0])
tvg(to_save)

np.savez('/Users/jkim/Desktop/mg2hk/to_send/data4PChIRIS2_{}.npz'.format(obsid), data = to_save, variables = ['iris_map_2794', 'aia_1600', 'aia_1700', 'aia_304', 'aia_193', 'aia_171', 'T (-5.2)', 'T (-4.2)', 'T (-3.2)', 'vlos (-5.2)', 'vlos (-4.2)', 'vlos (-3.2)', 'vturb (-5.2)', 'vturb (-4.2)', 'vturb (-3.2)', 'log(nne) (-5.2)', 'log(nne) (-4.2)', 'log(nne) (-3.2)', 'unc_T (-5.2)', 'unc_T (-4.2)', 'unc_T (-3.2)', 'unc_vlos (-5.2)', 'unc_vlos (-4.2)', 'unc_vlos (-3.2)', 'unc_vturb (-5.2)', 'unc_vturb (-4.2)', 'unc_vturb (-3.2)', 'unc_log(nne) (-5.2)', 'unc_log(nne) (-4.2)', 'unc_log(nne) (-3.2)'], mask = m, normalizing_factor = normalizing_factor)

#with open('/Users/jkim/Desktop/mg2hk/to_send/aia2iris_iris_l2_{}.npy'.format(obsid), 'wb') as f:
#	np.save(f, to_save)

#to inspect
