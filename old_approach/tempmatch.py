import cv2
import numpy as np
from matplotlib import pyplot as plt

aia_align_path = "/Users/jkim/Desktop/mg2hk/ex_aia.png"
raster_align_path = "/Users/jkim/Desktop/mg2hk/output/raster_to_align.png"

img_rgb = cv2.imread(aia_align_path)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(raster_align_path, 0)

width, height = template.shape[0], template.shape[1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
#plt.imshow(res, cmap='gray')

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + width, top_left[1] + height)
cv2.rectangle(img_rgb, top_left, bottom_right, (0, 0, 255), 2) 

cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()