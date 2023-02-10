import pick_from_LMSAL
import my_fits
import numpy as np
import rebin

from iris_lmsalpy import extract_irisL2data as ei

def new_load(obsid):
	numraster = 0
	print("testing with: (OBSID - {})".format(obsid))
	try:
		iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=numraster)
		aia_file = pick_from_LMSAL.obsid_raster2aia(obsid)

	except:
		print("error")
		exit(0)

	print("1", iris_file)
	print("2", aia_file)

	hdr_iris_data = ei.only_header(iris_file[0])
	aux_hdr_iris_data = ei.only_header(iris_file[0], extension=1)
	xcen_iris = hdr_iris_data['XCEN']
	ycen_iris = hdr_iris_data['YCEN']
	xscl_iris = aux_hdr_iris_data['CDELT3']
	yscl_iris = aux_hdr_iris_data['CDELT2']

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

new_load('20221026_041953_3600609177')

#pick_from_LMSAL.obsid_raster2aia('20221026_041953_3600609177')

# my_fits.read('/System/Volumes/Data/links/irisa/data/level2/2022/10/26/20221026_041953_3600609177/aia')