# creating + saving images for the randomly sample data to separate those w/ artifacts
import os
import numpy as np
import matplotlib.pyplot as plt

folder_path = '/Users/jkim/Desktop/mg2hk/sample_data_temp'
to_save_path = '/Users/jkim/Desktop/mg2hk/to_look/'

for data_cube_path in os.listdir(folder_path):
	obsid = data_cube_path[2:28]
	full_path = folder_path+'/'+data_cube_path
	data_cube = np.load(full_path)
	transposed_image_arrays = np.transpose(data_cube['data'], [2, 0, 1])

	#displaying images
	fig, ax = plt.subplots(1, 3, figsize = [20, 8])
	ax[0].imshow(transposed_image_arrays[0]) #iris_map
	ax[1].imshow(transposed_image_arrays[1]) #aia_1600
	ax[2].imshow(transposed_image_arrays[6]) #temperature (-5.2)

	#labelling images
	ax[0].title.set_text(data_cube['variables'][0]) #iris_map
	ax[1].title.set_text(data_cube['variables'][1]) #aia_1600
	ax[2].title.set_text(data_cube['variables'][6]) #temperature (-5.2)
	fig.suptitle("OBSID: "+obsid)
	
	#saving image
	plt.savefig(to_save_path+'visualized'+obsid+'.png')

	plt.close()