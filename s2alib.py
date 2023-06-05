import numpy as np
import cv2 as cv

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

class sji2aia:
	def __init__(self, sji_img, iris_t, i_aia):
		self.sji_img = sji_img
		self.iris_t = iris_t
		self.aia_i = i_aia

def template_match(main, aia_index, sji_img):
	method = eval('cv.TM_CCOEFF_NORMED')
	# aia_img = main[0][aia_index, :,:].astype(np.uint8)
	aia_img = main[aia_index, :,:].astype(np.uint8)
	res = cv.matchTemplate(sji_img.astype(np.uint8), aia_img, method)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

	if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
		top_left = min_loc
	else:
		top_left = max_loc

	#bottom_right = (top_left[0] + sji_img.shape[1], top_left[1] + sji_img.shape[0])
	# cv.rectangle(aia_img, top_left, bottom_right, (255,0,0), 2)
	# plt.subplot(131), plt.imshow(res, cmap='gray')
	# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	# plt.subplot(132), plt.imshow(aia_img.astype(np.uint8), cmap='gray')
	# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	# plt.subplot(133), plt.imshow(sji_img, cmap='gray')
	# plt.show()

	cx, cy = top_left[0] + sji_img.shape[1]/2, top_left[1]+sji_img.shape[0]/2

	return cx, cy