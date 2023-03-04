# coding: utf-8
import pick_from_LMSAL
import my_fits
from iris_lmsalpy import extract_irisL2data as ei

import rebin
obsid = '20230105_153122_3664103603'
pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')
path = pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')[0]
aia_main = my_fits.read(path, ext=0)
aia_extra = my_fits.read(path, ext=1)
iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=0)[0]
iris_raster = ei.load(iris_file)
aia_time2iris = aia_extra[0][:,0]
start_iris2aia = np.argmin(np.abs(aia_extra[0][:,0]))
iris_t_s = iris_raster.raster['Mg II k 2796'].date_time_acq_in_seconds
end_iris2aia =  np.argmin(np.abs(aia_extra[0][:,0]-iris_t_s[-1]))
iris_xcenix = iris_raster.raster['Mg II k 2796'].XCENIX
t_iris = iris_raster.raster['Mg II k 2796']['date_time_acq_in_seconds']


interp = []
class iris2aia:
	def __init__(self, t_iris, x_iris, i_aia):
		self.iris_time = t_iris
		self.iris_x = x_iris
		self.aia_time_index = i_aia

	def add_iris_x(self, new_iris_x):
		self.new_iris_x = new_iris_x
        
def closest_time(iris_time, aia_times):
	differences = [abs(iris_time - aia_time) for aia_time in aia_times]
	# print("time difference: ", min(differences))
	return differences.index(min(differences))
    
for i, it in enumerate(t_iris):
	ob = iris2aia(it, iris_xcenix[i], closest_time(it, aia_time2iris))
	interp.append(ob)
    
len(interp)
len(t_iris)
len(aia_time2iris)

final_raster = []

for i, iris_obs in enumerate(interp):
	if aia_time2iris[iris_obs.aia_time_index] > iris_obs.iris_time:
		f_iris_ind = list(map(lambda k: k > aia_time2iris[iris_obs.aia_time_index], t_iris)).index(True)
		dt = interp[f_iris_ind].iris_time - iris_obs.iris_time
		dx = interp[f_iris_ind].iris_x - iris_obs.iris_x

		new_iris_x = iris_obs.iris_x + (aia_time2iris[iris_obs.aia_time_index]-iris_obs.iris_time)*dx/dt

	elif aia_time2iris[iris_obs.aia_time_index] < iris_obs.iris_time:
		i_iris_ind = [i for i, x in enumerate(t_iris-aia_time2iris[iris_obs.aia_time_index]) if x < 0][-1]

		dt = iris_obs.iris_time - interp[i_iris_ind].iris_time
		dx = iris_obs.iris_x - interp[i_iris_ind].iris_x

		new_iris_x = interp[i_iris_ind].iris_x + (aia_time2iris[iris_obs.aia_time_index]-interp[i-1].iris_time)*dx/dt

	else:
		print("are you sure??")

	interp[i].add_iris_x(new_iris_x)
	raster_col = aia_main[0][i,:,round(iris_obs.new_iris_x):round(iris_obs.new_iris_x)+1]
	final_raster.append(raster_col)
