from matplotlib.pyplot import figure, draw, pause
from iris_lmsalpy import extract_irisL2data as ei

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import rebin
import os
import numpy as np

datapath = os.getcwd() + "/data/"

raster_folder = datapath + "iris_l2_20220607_202829_3620106067_raster/"
aia_aligned_path = "/Users/jkim/Desktop/mg2hk/output/aligned_color.png"
raster_path = raster_folder + "iris_l2_20220607_202829_3620106067_raster_t000_r00000.fits"



iris_raster = ei.load(raster_path)
mgii = iris_raster.raster['Mg II k 2796'].data[:,:,28]

aligned = mpimg.imread(aia_aligned_path)


dims = [404, 230]

x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')

for phase in np.linspace(0, 10*np.pi, 100):
	line1.set_ydata(np.sin(0.5 * x + phase))
	fig.canvas.draw()
	fig.canvas.flush_events()

