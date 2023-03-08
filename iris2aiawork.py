# coding: utf-8
import pick_from_LMSAL
import my_fits
import numpy as np
from iris_lmsalpy import extract_irisL2data as ei

import rebin
# obsid = '20230105_153122_3664103603'
obsid = '20230103_194208_3610108077'
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


interp = []
class iris2aia:
	def __init__(self, t_iris, x_iris, i_aia, y_iris):
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
	interp.append(ob)


h_iris_arcsec = extent_iris_arcsec[3]
col_h_px = h_iris_arcsec/aia_yscl
aia_mid_y_px = aia_main[0].shape[1]/2

raster = np.zeros((int(h_iris_arcsec/aia_yscl), len(interp)))

for i, iris_obs in enumerate(interp):
	closest_aia_time = aia_time2iris[iris_obs.aia_time_index]
	if closest_aia_time > iris_obs.iris_time:
		f_iris_ind = list(map(lambda k: k > closest_aia_time, t_iris)).index(True)
		dt = interp[f_iris_ind].iris_time - iris_obs.iris_time
		dx = interp[f_iris_ind].iris_x - iris_obs.iris_x

		new_iris_x = iris_obs.iris_x + (closest_aia_time-iris_obs.iris_time)*dx/dt

	elif aia_time2iris[iris_obs.aia_time_index] < iris_obs.iris_time:
		i_iris_ind = [l for l, x in enumerate(t_iris-closest_aia_time) if x < 0][-1]

		dt = iris_obs.iris_time - interp[i_iris_ind].iris_time
		dx = iris_obs.iris_x - interp[i_iris_ind].iris_x

		new_iris_x = interp[i_iris_ind].iris_x + (closest_aia_time-interp[i-1].iris_time)*dx/dt

	else:
		print("are you sure??")


	interp[i].add_iris_x(new_iris_x)
	coi = (iris_obs.new_iris_x - aia_xcen) / aia_xscl + aia_main[0].shape[2] / 2

	raster_col = aia_main[0][iris_obs.aia_time_index,round(aia_mid_y_px-col_h_px/2):round(aia_mid_y_px+col_h_px/2),round(coi):round(coi)+1].ravel()
#	raster_col = aia_main[0][iris_obs.aia_time_index, :, round(coi):round(coi) + 1].ravel()
	raster[:,i] = raster_col