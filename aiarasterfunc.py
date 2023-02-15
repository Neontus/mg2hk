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

iris_filename = "/Users/jkim/Desktop/mg2hk/data/iris_l2_20230103_194208_3610108077_raster_t000_r00000.fits"
aia_folder = "/Users/jkim/Desktop/mg2hk/data/iris_l2_20230103_194208_3610108077_SDO/"

def closest_time(iris_time, aia_times):
	differences = [abs(parse(iris_time) - parse(aia_time)) for aia_time in aia_times]
	# print("time difference: ", min(differences))
	return differences.index(min(differences))

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

	aia_times = []
	aia_imgs = []

	for aia_file in os.listdir(aia_folder):
		aia_img, aia_header = my_fits.read(aia_folder+aia_file)
		aia_time = aia_header['DATE_OBS']
		aia_times.append(aia_time)
		aia_imgs.append(aia_img)

	time_dict = dict(zip(iris_times, [closest_time(t, aia_times) for t in iris_times]))

	for i, t in enumerate(time_dict):
		aia = aia_imgs[i]




new_load()




#pick_from_LMSAL.obsid_raster2aia('20221026_041953_3600609177')

# my_fits.read('/System/Volumes/Data/links/irisa/data/level2/2022/10/26/20221026_041953_3600609177/aia')