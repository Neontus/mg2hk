import numpy as np
import alignlib
import pandas as pd
import time
import cv2
import matplotlib.pyplot as plt
import re
import os
import saveblank
from iris_lmsalpy import find


to_test = ['20220924_021937_3620110077',
 '20221006_135137_3620108077',
 '20221007_050103_3620108077',
 '20221007_135202_3620108077',
 '20221007_211016_3620108077',
 '20221008_030317_3620108077',
 '20221025_195518_3600609177',
 '20221026_071530_3600609177',
 '20221026_170654_3600609177',
 '20221029_065819_3600609177']



for o in to_test:
	start = time.time()

	iris, aia = alignlib.load(o)
	a, i, b = 0.214265, 0.241511, 25.420002
	align = alignlib.super_align(aia, iris, a, i, b)
	res = align.evolve()

	end = time.time()
	runtime = end - start

	print("OBSID: ", o, "\n Runtime: ", runtime, "\n Result:", res['fun'])

	a2, i2, b2 = res['x'][0], res['x'][1], res['x'][2]

	AIA_THRESH = alignlib.get_top_n(aia, a2)
	IRIS_THRESH_L = alignlib.get_top_n(iris, i2)
	IRIS_THRESH_H = 450.025

	aia_to_align = ((aia > AIA_THRESH) * 255).astype(np.uint8)
	iris_to_align = cv2.normalize(alignlib.lee_filter((alignlib.imgthr(iris, IRIS_THRESH_L, IRIS_THRESH_H) * 255), b2), None, 0,
	                              255, cv2.NORM_MINMAX).astype('uint8')

	matrix, walign, halign = alignlib.sift_ransac(aia_to_align, iris_to_align, debug=False)
	aligned_color_aia = cv2.warpAffine(aia, matrix, (walign, halign))

	fig, ax = plt.subplots(1, 3, figsize=[15, 10])
	ax[0].imshow(aia, origin='lower', cmap='afmhot',
	             interpolation=None)  # , cmap=cm.sdoaia1600)
	# saveblank('iris_to_align')
	ax[1].imshow(aligned_color_aia, origin='lower',
	             cmap='afmhot')
	ax[2].imshow(iris, origin='lower', cmap='afmhot')
	plt.show()
	saveblank.saveblank("testdata/",o)