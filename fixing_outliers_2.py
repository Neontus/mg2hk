#this code is commented w/ verbose variables names as the function
# clean_outliers() in s2alib.py

#code assumes data is in dataset of datacubes w/ all layers + obsids of those w/ photon beams or other artifacts

import os
import numpy as np

outlier_obsids = [path[7:33] for path in os.listdir("/Users/jkim/Desktop/mg2hk/fixed_dataset_1/artifacts/")]
data_cube_directory = "/Users/jkim/Desktop/mg2hk/dataset/"
dir_to_save = "/Users/jkim/Desktop/mg2hk/fixed_dataset_1/cleaned/"

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
	q1, q3 = np.quantile(unn_iris.flatten(), [0.25, 0.85])
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