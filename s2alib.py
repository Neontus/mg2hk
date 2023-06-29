import pick_from_LMSAL
import my_fits
import numpy as np
import cv2 as cv
import rebin
import matplotlib.pyplot as plt
from iris_lmsalpy import extract_irisL2data as ei
from iris_lmsalpy import iris2
from scipy import interpolate, signal
from alignlib import avg_diff


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
	return differences.index(min(differences))

class sji2aia:
	def __init__(self, sji_img, iris_t, i_aia):
		self.sji_img = sji_img
		self.iris_t = iris_t
		self.aia_i = i_aia

def template_match(main, aia_index, sji_img):
	method = eval('cv.TM_CCOEFF_NORMED')
	aia_img = main[aia_index, :,:].astype(np.uint8)
	res = cv.matchTemplate(sji_img.astype(np.uint8), aia_img, method)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

	if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
		top_left = min_loc
	else:
		top_left = max_loc

	cx, cy = top_left[0] + sji_img.shape[1]/2, top_left[1]+sji_img.shape[0]/2

	return cx, cy

def s2adatacubeassembly(obsid, dir_to_save):
	#loading aia + aia variables
	aia_1600_path = pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')[0]
	aia_main = my_fits.read(aia_1600_path, ext=0)
	aia_extra = my_fits.read(aia_1600_path, ext=1)
	aia_time2iris = aia_extra[0][:,0] #time of aia images relative to iris obs
	aia_xscl = aia_main[1]['CDELT1'] # AIA x-scale of "/x
	aia_yscl = aia_main[1]['CDELT2'] # AIA y-scale of "/x

	#loading iris + iris variables
	iris_path = pick_from_LMSAL.obsid_raster(obsid, raster=0)[0]
	iris_raster = ei.load(iris_path, radcal = 'CGS_NU') #try on samples
	t_iris = iris_raster.raster['Mg II k 2796'].date_time_acq_in_seconds # time of each slit formatted in sec since begin of obs.
	iris_xcenix = iris_raster.raster['Mg II k 2796'].XCENIX # center x-coord of slit
	iris_ycenix = iris_raster.raster['Mg II k 2796'].YCENIX # center y-coord slit
	iris_xscl = iris_raster.raster['Mg II k 2796'].SPXSCL # IRIS x-scale of "/x
	iris_yscl = iris_raster.raster['Mg II k 2796'].SPYSCL # IRIS y-scale of "/y
	wl_iris = iris_raster.raster['Mg II k 2796'].wl # IRIS wavelengths
	pos = ei.get_ori_val(wl_iris, 2794.) # Closest to 2794
	iris_map = iris_raster.raster['Mg II k 2796'].data[:,:,pos] # IRIS map at 2794
	extent_iris_arcsec = iris_raster.raster['Mg II k 2796'].extent_arcsec_arcsec #extent of IRIS in "
	n_rasters = len(pick_from_LMSAL.obsid_raster(obsid, raster=-1)) #number of rasters available

	#loading sji + sji variables
	sji_path = pick_from_LMSAL.obsid_sji(obsid, pattern='1400')[0]
	sji = ei.load(sji_path)
	sji1400 = sji.SJI['SJI_1400'] # SJI data 1400
	slitxs = sji1400.SLTPX1IX # X pos. of center slit
	sji_yscl = sji1400.SPYSCL # SJI y-scale of "/y -- I think it is same as iris_yscl
	sji_xscl = sji1400.SPCSCL #SJI x-scale of "/x -- same as above
	aux_sji = ei.only_data(sji_path, extension=-2) # auxiliary SJI info
	sjit_t_s = aux_sji[:,0] #[::-1] # time of SJI slits
	
	#finding closest time in AIA for each IRIS/SJI slit and storing in list
	iris_2_aia_slits = []
	for i, it in enumerate(t_iris):
		slit_observation = iris2aia(it, iris_xcenix[i], closest_time(it, aia_time2iris), iris_ycenix[i])
		iris_2_aia_slits.append(slit_observation)

	# finding the center coordinate for each slit in the corresponding AIA image
	interp = []
	xcenters = []
	ycenters = []

	SJI_image_width = int((sji1400.data.shape[1]-(avg_diff(slitxs))*sji1400.data.shape[2])/2) # how wide relevant SJI image region is px
	l_iris =len(t_iris) # count of iris slits
	sji_period = l_iris/sji1400.data.shape[2] #how many slits between each SJI image = num of filters??

	scaled_shape_s2a = (sji1400.data.shape[0]*sji_yscl/aia_yscl, SJI_image_width*2*sji_xscl/aia_xscl) #sji image shape, scaled from sji scale to aia scale 0.6"/px
	aia_main_copy = aia_main[0].copy()

	for i in range(l_iris):
		closest_sji = closest_time(t_iris[i], sjit_t_s) # index of closest SJI image
		slit_x_coord = slitxs[closest_sji] # x-coord of slit
		rounded_slit_x_coord = round(slit_x_coord) # rounded x-coord of slit
		relevant_sji_image = sji1400.data[:,rounded_slit_x_coord-SJI_image_width:rounded_slit_x_coord+SJI_image_width,closest_sji]
		aia_index = ei.get_ori_val(aia_time2iris,t_iris[i]) #index of AIA image closest to time of slit
		relevant_sji_scaled = rebin.congrid(relevant_sji_image, scaled_shape_s2a) # slit is not at slit_x_coord anymore, now it is middle column, scaled to aia scale
		cenx, ceny = template_match(aia_main_copy, aia_index, relevant_sji_scaled) # center coordinates
		xcenters.append(cenx)
		ycenters.append(ceny)

	#interpolate center coordinates
	ycenters = np.array(ycenters)
	x = np.arange(0, l_iris, 1)
	f = interpolate.interp1d(x, ycenters, fill_value = "extrapolate")
	newycenters = f(np.arange(l_iris))
	newxcenters = xcenters

	#create raster
	raster = np.zeros((int(scaled_shape_s2a[0]), l_iris)) # aia scale y, count of iris px x
	for i, (xcen, ycen) in enumerate(zip(newxcenters, newycenters)):
		aia_i = iris_2_aia_slits[i].aia_time_index
		scy = scaled_shape_s2a[0]/2
		slit = aia_main[0][aia_i, int(ycen-scy):int(ycen-scy)+int(scaled_shape_s2a[0]), int(xcen):int(xcen)+1].flatten()
		raster[:,i] = slit

	#scale raster + iris_map to final shape in AIA scale
	final_s2a_shape = (scaled_shape_s2a[0], abs(iris_map.shape[1]*iris_xscl/aia_xscl)) #final scaled shape in aia scale
	final_raster = rebin.congrid(raster, final_s2a_shape)
	final_iris = rebin.congrid(iris_map, final_s2a_shape)

	#replicating for other channels
	channels = ['1700', '304', '193', '171']
	paths = [pick_from_LMSAL.obsid_raster2aia(obsid, pattern=p)[0] for p in channels]
	mains = [my_fits.read(p, ext=0) for p in paths] #main aia files
	extras = [my_fits.read(p, ext=1) for p in paths] #extra aia files
	t2is = [ex[0][:,0] for ex in extras] #time relative to iris
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

	#IRIS2 Inversions	
	output = iris2.invert(iris_path)
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

	inv_stack = np.concatenate([temp, vlos, vturb, nne, unc_temp, unc_vlos, unc_vturb, unc_nne, mask], axis=2)
	final_inv = [] 
	for i in range(inv_stack.shape[2]):
		final_inv.append(rebin.congrid(inv_stack[:,:,i], final_s2a_shape))

	data_cube = [final_iris, final_raster] + rasters + final_inv

	scaled_mask = rebin.congrid(mask, final_s2a_shape)>0.1
	where_mask = np.where(scaled_mask != 0)
	masked = [array[where_mask[0][0]:where_mask[0][-1], where_mask[1][0]:where_mask[1][-1]] for array in data_cube]

	normalizing_factor = [np.nanmax(x[:,:]) for x in masked]
	normalized = [arr/normalizing_factor[i]*255 for i,arr in enumerate(masked)]

	to_save = np.transpose(np.stack(tuple(normalized)), [1, 2, 0]) #for use with tvg
	#to use w/ matplotlib, remove or transpose to inverse: [2, 0, 1]
	variables = ['iris_map_2794', 'aia_1600', 'aia_1700', 'aia_304', 'aia_193', 'aia_171', 'T (-5.2)', 'T (-4.2)', 'T (-3.2)', 'vlos (-5.2)', 'vlos (-4.2)', 'vlos (-3.2)', 'vturb (-5.2)', 'vturb (-4.2)', 'vturb (-3.2)', 'log(nne) (-5.2)', 'log(nne) (-4.2)', 'log(nne) (-3.2)', 'unc_T (-5.2)', 'unc_T (-4.2)', 'unc_T (-3.2)', 'unc_vlos (-5.2)', 'unc_vlos (-4.2)', 'unc_vlos (-3.2)', 'unc_vturb (-5.2)', 'unc_vturb (-4.2)', 'unc_vturb (-3.2)', 'unc_log(nne) (-5.2)', 'unc_log(nne) (-4.2)', 'unc_log(nne) (-3.2)']
	outpath = dir_to_save +'/data4PChIRIS2_{}.npz'.format(obsid)

	np.savez(outpath, data = to_save, variables = variables, mask = scaled_mask, normalizing_factor = normalizing_factor)
	print("saved at: ", outpath)

def clean_outliers(outlier_obsids, data_cube_directory, dir_to_save):
	for obsid in outlier_obsids:
		data_cube = np.load("{}data4PChIRIS2_{}.npz".format(data_cube_directory, obsid))
		iris_map = data_cube['data'][:,:,0]
		aia_1600 = data_cube['data'][:,:,1]
		aia_1700 = data_cube['data'][:,:,2]
		aia_304 = data_cube['data'][:,:,3]
		temp_5_2 = data_cube['data'][:,:,6]
		rel_layers = [aia_1600, aia_1700, aia_304, temp_5_2]
		nf = [data_cube['normalizing_factor'][ind] for ind in [0, 1, 2, 3, 6]]
		
		unn_iris = iris_map*nf[0]
		q1, q3 = np.quantile(unn_iris.flatten(), [0.25, 0.75])
		iqr = q3-q1
		ub = q3+iqr
		outlier_mask = np.where(unn_iris>=ub)

		output_layers = []
		output_nf = []

		for i, layer in enumerate(rel_layers):
			unn_arr = layer*nf[i+1]
			unn_arr_w_o = unn_arr.copy(); unn_arr_w_o[outlier_mask] = 0
			avg_w_o = np.mean(unn_arr_w_o)
			unn_arr_masked = unn_arr.copy(); unn_arr_masked[outlier_mask] = avg_w_o
			normalizing_factor_masked = np.nanmax(unn_arr_masked)
			normalized_masked = unn_arr_masked/normalizing_factor_masked*255
			output_layers.append(normalized_masked)
			output_nf.append(normalizing_factor_masked)

		x_stack = np.stack(tuple(output_layers[0:3]))
		y = output_layers[3]

		np.savez('{}xydata_{}.npz'.format(dir_to_save, obsid), x = x_stack, y = y, variables = ['x data - 1600, 1700, 304', 'y data - temp @ -5.2'], normalizing_factor = output_nf)
		print("saved")

def prep_clean(clean_obsids, data_cube_directory, dir_to_save):
	for obsid in clean_obsids:
		data_cube = np.load("{}data4PChIRIS2_{}.npz".format(data_cube_directory, obsid))
		aia_1600 = data_cube['data'][:,:,1]
		aia_1700 = data_cube['data'][:,:,2]
		aia_304 = data_cube['data'][:,:,3]
		temp_5_2 = data_cube['data'][:,:,6]
		nf = [data_cube['normalizing_factor'][ind] for ind in [1, 2, 3, 6]]

		x_stack = np.stack(tuple([aia_1600, aia_1700, aia_304]))
		y = temp_5_2

		np.savez('{}xydata_{}.npz'.format(dir_to_save, obsid), x = x_stack, y = y, variables = ['x data - 1600, 1700, 304', 'y data - temp @ -5.2'], normalizing_factor = nf)
		print("saved")

def prep_clean_data_cubes(clean_obsids, data_cube_directory, dir_to_save):
	for obsid in clean_obsids:
		data_cube = np.load("{}data4PChIRIS2_{}.npz".format(data_cube_directory, obsid))
		aia_1600 = data_cube['data'][:,:,1]
		aia_1700 = data_cube['data'][:,:,2]
		aia_304 = data_cube['data'][:,:,3]
		aia_193 = data_cube['data'][:,:,4]
		aia_171 = data_cube['data'][:,:,5]
		temp_5_2 = data_cube['data'][:,:,6]
		nf = [data_cube['normalizing_factor'][ind] for ind in [1, 2, 3, 4, 5, 6]]

		x_stack = np.stack(tuple([aia_1600, aia_1700, aia_304, aia_193, aia_171]))
		y = temp_5_2

		np.savez('{}xydata_{}.npz'.format(dir_to_save, obsid), x = x_stack, y = y, variables = ['x data - 1600, 1700, 304, 193, 171', 'y data - temp @ -5.2'], normalizing_factor = nf)
		print("saved")


def checkpoint(message, variable):
	print("_"*10+"CHECKPOINT"+"_"*10)
	print(message)
	print("VARIABLE NAME: ", f'{variable=}'.split('=')[0])
	print("data: ", variable)
	print("type: ", type(variable))
	print("dtype: ", variable.dtype)
	print("_"*10+"END CHECKPOINT"+"_"*10)