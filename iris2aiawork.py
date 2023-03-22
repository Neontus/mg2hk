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
from alignlib import falign



# obsid = '20230105_153122_3664103603'
obsid = '20230103_194208_3610108077'
correlate = False

pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')
path = pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')[0]
aia_main = my_fits.read(path, ext=0)
aia_extra = my_fits.read(path, ext=1)
iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=0)[0]
iris_raster = ei.load(iris_file)
hiris, wiris = iris_raster.raster['Mg II k 2796']['extent_heliox_helioy'][3]-iris_raster.raster['Mg II k 2796']['extent_heliox_helioy'][2], iris_raster.raster['Mg II k 2796']['extent_heliox_helioy'][1]-iris_raster.raster['Mg II k 2796']['extent_heliox_helioy'][0]
aia_time2iris = aia_extra[0][:,0]
start_iris2aia = np.argmin(np.abs(aia_extra[0][:,0]))
iris_t_s = iris_raster.raster['Mg II k 2796'].date_time_acq_in_seconds
end_iris2aia =  np.argmin(np.abs(aia_extra[0][:,0]-iris_t_s[-1]))
iris_xcenix = iris_raster.raster['Mg II k 2796'].XCENIX
iris_ycenix = iris_raster.raster['Mg II k 2796'].YCENIX
t_iris = iris_raster.raster['Mg II k 2796']['date_time_acq_in_seconds']
aia_xcen = aia_main[1]['XCEN']
aia_xscl = aia_main[1]['CDELT1']
aia_yscl = aia_main[1]['CDELT2']
extent_iris_arcsec = iris_raster.raster['Mg II k 2796'].extent_arcsec_arcsec
iris_xscl = iris_raster.raster['Mg II k 2796'].SPXSCL
iris_yscl = iris_raster.raster['Mg II k 2796'].SPYSCL
iris_ycen = iris_raster.raster['Mg II k 2796'].YCEN
wl_iris = iris_raster.raster['Mg II k 2796'].wl
pos = ei.get_ori_val(wl_iris, 2794.)
iris_map = iris_raster.raster['Mg II k 2796'].data[:,:,pos]


interp = []
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


offset = [3, 3, 5, 5] #xr, xl, yd, yu
xr, xl, yd, yu = offset
pxl, pxr, pyd, pyu = round(xl/aia_xscl), round(xr/aia_xscl), round(yd/aia_yscl), round(yu/aia_yscl)

for i, it in enumerate(t_iris):
	ob = iris2aia(it, iris_xcenix[i], closest_time(it, aia_time2iris), iris_ycenix[i])
	interp.append(ob)


interf = interpolate.interp1d(t_iris, iris_xcenix, fill_value='extrapolate')
intery = interpolate.interp1d(t_iris, iris_ycenix, fill_value='extrapolate')
adt = (t_iris[-1]-t_iris[0])/len(t_iris)

xl_in_t = np.arange(-1*xl*adt, 0, adt)
xl_obs = []
for t in xl_in_t:
	xl_obs.append(iris2aia(t, interf(t)*1, closest_time(t, aia_time2iris), intery(t)*1))

xr_in_t = np.arange(t_iris[-1], t_iris[-1]+xr*adt, adt)
xr_obs = []
for t in xr_in_t:
	xr_obs.append(iris2aia(t, interf(t)*1, closest_time(t, aia_time2iris), intery(t)*1))

interp = xl_obs + interp + xr_obs
new_t_iris = np.concatenate((xl_in_t, t_iris, xr_in_t), axis=0)

h_iris_arcsec = extent_iris_arcsec[3] + yu + yd
col_h_px = h_iris_arcsec/aia_yscl
aia_mid_y_px = (aia_main[0].shape[1]+pyu-pyd)/2

raster = np.zeros((round(h_iris_arcsec/aia_yscl), len(interp)))
#raster = np.zeros((aia_main[0].shape[1], len(interp)))

for i, iris_obs in enumerate(interp):
	closest_aia_time = aia_time2iris[iris_obs.aia_time_index]
	if closest_aia_time > iris_obs.iris_time:
		f_iris_ind = list(map(lambda k: k > closest_aia_time, new_t_iris)).index(True)
		dt = interp[f_iris_ind].iris_time - iris_obs.iris_time
		dx = interp[f_iris_ind].iris_x - iris_obs.iris_x

		new_iris_x = iris_obs.iris_x + (closest_aia_time-iris_obs.iris_time)*dx/dt

	elif closest_aia_time < iris_obs.iris_time:
		i_iris_ind = [l for l, x in enumerate(new_t_iris-closest_aia_time) if x < 0][-1]

		dt = iris_obs.iris_time - interp[i_iris_ind].iris_time
		dx = iris_obs.iris_x - interp[i_iris_ind].iris_x

		new_iris_x = interp[i_iris_ind].iris_x + (closest_aia_time-interp[i-1].iris_time)*dx/dt

	else:
		print("are you sure??")


	interp[i].add_iris_x(new_iris_x)
	coi = (iris_obs.new_iris_x - aia_xcen) / aia_xscl + aia_main[0].shape[2] / 2

	#raster_col = aia_main[0][iris_obs.aia_time_index,round(aia_mid_y_px-col_h_px/2):round(aia_mid_y_px+col_h_px/2),round(coi):round(coi)+1].ravel()
	raster_col = aia_main[0][iris_obs.aia_time_index,:,round(coi):round(coi) + 1].ravel()
	#	raster_col = aia_main[0][iris_obs.aia_time_index, :, round(coi):round(coi) + 1].ravel()

	if correlate:
		x = signal.resample(iris_map[:, i], round(len(iris_map) * iris_yscl / aia_yscl))
		y = raster_col
		correlation = signal.correlate(x, y, mode="full")
		lags = signal.correlation_lags(x.size, y.size, mode="full")
		lag = lags[np.argmax(correlation)]

		fraster_col = raster_col[-lag:round(col_h_px)-lag-1]

	else:
		# dy_arc = iris_obs.iris_y-iris_ycen
		# dy_px = dy_arc/aia_yscl
		# fraster_col = aia_main[0][
		# 	iris_obs.aia_time_index
		# 	,round(aia_mid_y_px - col_h_px / 2) : round(aia_mid_y_px + col_h_px / 2)
		#     ,round(coi):round(coi)+1
		# ].ravel()



		#taking middle of aia picture (old approach)
		fraster_col = aia_main[0][iris_obs.aia_time_index,round(aia_mid_y_px - col_h_px / 2):round(aia_mid_y_px + col_h_px / 2),round(coi):round(coi) + 1].ravel()

	raster[:,i] = fraster_col

# displaying raster + map
plt.ion()

#fig, ax = plt.subplots(1,2, figsize=[10,8], sharey=True, sharex=True)
#ax[1].imshow(iris_map, vmin=90, vmax=300, origin='lower', extent=extent_iris_arcsec); ax[1].set_title('iris map')
extent_aia_arc = [-1*yd, extent_iris_arcsec[1]+yu, -1*xl, raster.shape[0]*aia_main[1]['CDELT2']+xr]
#ax[0].imshow(raster, origin='lower', extent=extent_aia_arc); ax[0].set_title('synthetic aia raster')

#masking iris_map
#plt.imshow(np.ma.masked_where(np.invert(mask), iris_map), vmin=90, vmax=300, origin='lower', extent=extent_iris_arcsec)
mask = iris_raster.raster['Mg II k 2796'].mask
masked_iris_map = np.ma.masked_where(np.invert(mask), iris_map)
fin_iris_map = masked_iris_map[~np.isnan(masked_iris_map).all(axis=1)]
#masked_extent =

#scale iris array to same as aia using scales
h_aia_arc = extent_aia_arc[3]-extent_aia_arc[2]
new_h_iris_arc = fin_iris_map.shape[0]*h_aia_arc/raster.shape[1]
new_iris_shape = (new_h_iris_arc, fin_iris_map.shape[1])
iris_to_align = rebin.congrid(fin_iris_map, (new_h_iris_arc, fin_iris_map.shape[1]))

#align the two images
ob = falign(raster, iris_to_align)
fa, mat, err = ob.coalign()

#display
fig, ax = plt.subplots(1,3, figsize=[10,8], sharey=True, sharex=True)
ax[0].imshow(raster, origin='lower'); ax[0].set_title('synthetic aia raster')
ax[1].imshow(fa, origin='lower'); ax[1].set_title('aligned synthetic raster')
ax[2].imshow(iris_to_align, vmin=90, vmax=300, origin='lower'); ax[2].set_title('iris raster')

#result: iris-like aia raster cropped to iris