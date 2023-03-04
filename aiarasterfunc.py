import pick_from_LMSAL
import my_fits
import numpy as np
import rebin
import os

from dateutil.parser import parse
from iris_lmsalpy import extract_irisL2data as ei

#from iris_lmsalpy import hcr2fits
#query_text = 'https://www.lmsal.com/hek/hcr?cmd=search-events3&outputformat=json&startTime=2013-07-20T00:00&stopTime=2023-01-20T00:00&minnumRasterSteps=100&hasData=true&minxCen=-400&maxxCen=400&minyCen=-400&maxyCen=400&hideMostLimbScans=true&limit=200'
#list_urls = hcr2fits.get_fits(query_text)

obsid = '20230103_194208_3610108077'

# iris_filename = "/Users/jkim/Desktop/mg2hk/data/iris_l2_20230103_194208_3610108077_raster_t000_r00000.fits"
# aia_folder = "/Users/jkim/Desktop/mg2hk/data/iris_l2_20230103_194208_3610108077_SDO/"

iris_filename = pick_from_LMSAL.obsid_raster(obsid)[0]
aia_file = pick_from_LMSAL.obsid_raster2aia(obsid, '335')[0]
# change to 1600


def closest_time(iris_time, aia_times):
	differences = [abs(iris_time - aia_time) for aia_time in aia_times]
	# print("time difference: ", min(differences))
	return differences.index(min(differences))

class iris2aia:
	def __init__(self, t_iris, x_iris, i_aia):
		self.iris_time = t_iris
		self.iris_x = x_iris
		self.aia_time_index = i_aia

	# find where IRIS slit would be at aia_time
	# capture this column from aia image, put into final raster


# for i, it in enumerate(iris_t_s):
# 	ob = iris2aia(it, stepx[i], closest_time(it, aia_time2iris))
# 	interp.append(ob)


class aia2iris:
	def __init__(self, t_aia, x_aia, t_iris):


	def calc_adjusted(self, ):
		# if current iris x is before aia time

		dx = fx-ix #difference in iris xs before and after aia_time
		dt1 = ft-it #difference in iris ts before and after aia time
		dt2 = at-it #differnece between aia time and iris time (aia_time > iris_time)
		new_iris_x = self.iris_x

	# for each aia image, generate heliox range
	#  heliox_aia = (np.arange(aia_header['NAXIS1'])-  (aia_header['NAXIS1']/2.))*aia_header['CDELT1'] + xceni[j]
	# this should become 2d array: x axis time, yaxis heliox
	# then, interpolate x axis to iris times
	# then, find closest heliox to iris cenix
	# then, take this value from AIA image and put into final raster


def new_load():
	numraster = 0
	#print("testing with: (OBSID - {})".format(obsid))
	# try:
	# 	iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=numraster)
	# 	aia_file = pick_from_LMSAL.obsid_raster2aia(obsid)
	#
	# except:
	# 	print("error")
	# 	exit(0)
	iris_file = iris_filename

	hdr_iris_data = ei.only_header(iris_file)
	aux_hdr_iris_data = ei.only_header(iris_file, extension=1)
	xcen_iris = hdr_iris_data['XCEN']
	ycen_iris = hdr_iris_data['YCEN']
	xscl_iris = aux_hdr_iris_data['CDELT3']
	yscl_iris = aux_hdr_iris_data['CDELT2']

	iris_raster = ei.load(iris_file)
	extent_hx_hy = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy

	stepx = iris_raster.raster['Mg II k 2796'].XCENIX
	stepy = iris_raster.raster['Mg II k 2796'].YCENIX
	iris_times = iris_raster.raster['Mg II k 2796'].date_time_acq_ok

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

	# print(extent_iris)

	# aia_data = []
	#
	# for aia_file in os.listdir(aia_folder):
	# 	aia_img, aia_header = my_fits.read(aia_folder+aia_file)
	# 	aia_time = aia_header['DATE_OBS']
	# 	aia_xcen, aia_ycen, aia_xfov, aia_yfov = aia_header['XCEN'], aia_header['YCEN'], aia_header['FOVX'], aia_header['FOVY']
	# 	low_x, high_x, low_y, high_y = aia_xcen - aia_xfov / 2, aia_xcen + aia_xfov / 2, aia_ycen - aia_yfov / 2, aia_ycen + aia_yfov / 2
	# 	extent_aia = [low_x, high_x, low_y, high_y]
	#
	# 	aia_data.append({
	# 		'time': aia_time,
	# 		'img': aia_img,
	# 		'extent': extent_aia
	# 	})

	t_iris = iris_raster.raster['Mg II k 2796']['date_time_acq_in_seconds']



	time_dict = dict(zip(iris_times, [closest_time(t, [d['time'] for d in aia_data]) for t in iris_times]))

	for i, t in enumerate(time_dict):
		aia = aia_data[time_dict[t]]
		padding = 20
		croph = hiris + padding
		dim_aia = aia['img'].shape
		slit = aia['img'][dim_aia[0]-1, yc-hiris:yc+hiris, :]



new_load()




#pick_from_LMSAL.obsid_raster2aia('20221026_041953_3600609177')

# my_fits.read('/System/Volumes/Data/links/irisa/data/level2/2022/10/26/20221026_041953_3600609177/aia')