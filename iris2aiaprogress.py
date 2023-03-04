# coding: utf-8
import pick_from_LMSAL
import my_fits
import rebin

obsid = '20230105_153122_3664103603'
pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')
reload(pick_from_LMSAL)
pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')
reload(pick_from_LMSAL)
pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')
aia_extra = my_fits.read('aia_l2_20230103_194208_3610108077_335.fits', ext=1)

aia_extra = my_fits.read('aia_l2_20230103_194208_3610108077_335.fits', ext=1)
path = /Users/jkim/Desktop/mg2hk/data/iris_l2_20230103_194208_3610108077_SDO/aia_l2_20230103_194208_3610108077_335.fits
path = "/Users/jkim/Desktop/mg2hk/data/iris_l2_20230103_194208_3610108077_SDO/aia_l2_20230103_194208_3610108077_335.fits"
aia_extra = my_fits.read(path, ext=1)
aia_time2iris = aia_extra[0][:,0]
aia_time2iris
aia_extra[0].keys
aia_extra
aia_extra[0]
start_iris2aia = np.argmin(np.abs(aia_extra[0][:,0]))
start_iris2aia
aia_time2iris[51]
heliox_aia= []
aia_main = my_fits.read(path, ext=0)
aia_main[0]
aia_main.keys
aia_main[1]
aia_main[1]['NAXIS3']
aia_main[1]['NAXIS3']-start_iris2aia
start_iris2aia
heliox_aia = (np.arange(aia_header['NAXIS1'])-  (aia_header['NAXIS1']/2.))*aia_header['CDELT1'] + xceni[j]
aia_header
aia_img, aia_header = my_fits.read(path)
heliox_aia = (np.arange(aia_header['NAXIS1'])-  (aia_header['NAXIS1']/2.))*aia_header['CDELT1'] + xceni[j]
iris_filename = "/Users/jkim/Desktop/mg2hk/data/iris_l2_20230103_194208_3610108077_raster_t000_r00000.fits"
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
xceni = stepx
heliox_aia = (np.arange(aia_header['NAXIS1'])-  (aia_header['NAXIS1']/2.))*aia_header['CDELT1'] + xceni[j]
j = start_iris2aia
get_ipython().run_line_magic('pinfo', 'j')
j
heliox_aia = (np.arange(aia_header['NAXIS1'])-  (aia_header['NAXIS1']/2.))*aia_header['CDELT1'] + xceni[j]
heliox_aia
heliox_aia.shape
get_ipython().run_line_magic('pinfo', 'heliox_aia')
obsid = '20230105_153122_3664103603'
pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')
reload(pick_from_LMSAL)
pick_from_LMSAL.obsid_raster2aia(obsid, pattern='1600')
aia_time2iris
reload(ei)
iris_file = iris_filename
iris_raster = ei.load(iris_file)
iris_raster = ei.load(iris_file)
iris_raster.quick_look()
from iris_lmsalpy import extract_irisL2data as ei
iris_raster = ei.load(iris_file)
iris_raster.quick_look
iris_raster.quick_look()
reload(ei)
iris_raster = ei.load(iris_file)
iris_raster.quick_look()
iris_raster.raster['Mg II k 2796']['extent_date_time_acq_in_seconds']
iris_raster.raster['Mg II k 2796']['date_time_acq_in_seconds']
iris_raster.quick_look()
for j, t in enumerate(data.raster['Mg II k 2796'].date_time_acq_in_seconds):
           print(t, data.raster['Mg II k 2796'].XCENIX[j])
           
data = iris_raster
for j, t in enumerate(data.raster['Mg II k 2796'].date_time_acq_in_seconds):
           print(t, data.raster['Mg II k 2796'].XCENIX[j])
           
iris_raster.quick_look()
iris_t_s = iris_raster.raster['Mg II k 2796'].date_time_acq_in_seconds
iris_xcenix = iris_raster.raster['Mg II k 2796'].XCENIX
plt.plot(iris_t_s, iris_xcenix, markers = "o")
plt.plot(iris_t_s, iris_xcenix, marker = "o")
for j, t in enumerate(data.raster['Mg II k 2796'].date_time_acq_in_seconds):
           print(t, data.raster['Mg II k 2796'].XCENIX[j])
           
start_iris2aia =  np.argmin(np.abs(aia_extra[0][:,0]))
start_iris2aia
end_iris2aia =  np.argmin(np.abs(aia_extra[0][:,-1]))
end_iris2aia
aia_extra[0]
end_iris2aia =  np.argmin(aia_extra[0][:,0]-iris_t_s[-1])
end_iris2aia
aia_extra[0][:,0]-iris_t_s[-1]
end_iris2aia =  np.argmin(np.abs(aia_extra[0][:,0]-iris_t_s[-1]))
end_iris2aia
aia_extra[0][:,0][end_iris2aia]
len(stepx)
def closest_time(iris_time, aia_times):
	differences = [abs(parse(iris_time) - parse(aia_time)) for aia_time in aia_times]
	# print("time difference: ", min(differences))
	return differences.index(min(differences))
    
for it in iris_t_s:
    print(closest_time(it, aia_time2iris))
    
def closest_time(iris_time, aia_times):
	differences = [abs(iris_time - aia_time) for aia_time in aia_times]
	# print("time difference: ", min(differences))
	return differences.index(min(differences))
    
for it in iris_t_s:
    print(closest_time(it, aia_time2iris))
    
start_iris2aia
end_iris2aia
t_iris
t_iris = iris_raster.raster['Mg II k 2796']['date_time_acq_in_seconds']
zip(t_iris, stepx)
print(zip(t_iris, stepx)[0])
print(zip(t_iris, stepx))
print(list(zip(t_iris, stepx)))
class iris_obs:
    def __init__(self, t_iris):
        self.time = t_iris
        
class iris_obs:
	def __init__(self, t_iris, x_iris, t_aia):
		self.time = t_iris
		self.x = x_iris
		self.aia_time = t_aia
        
o = iris_obs(t_iris[0], stepx[0], aia_time2iris)
o = iris_obs(t_iris[0], stepx[0], aia_time2iris[0])
o.time
o.x
o.aia_time
class iris2aia:
	def __init__(self, t_iris, x_iris, t_aia):
		self.time = t_iris
		self.x = x_iris
		self.aia_time = t_aia
        
interp = []
for i, it in enumerate(iris_t_s):
    ob = iris2aia(it, stepx, aia_time2iris[closest_time(it, aia_time2iris)])
    interp.append(ob)
    
interp[0]
interp[0].time, interp[0].x, interp[0].aia_time
for i, it in enumerate(iris_t_s):
    ob = iris2aia(it, stepx[i], aia_time2iris[closest_time(it, aia_time2iris)])
    interp.append(ob)
    
interp[0].time, interp[0].x, interp[0].aia_time
interp = []
for i, it in enumerate(iris_t_s):
    ob = iris2aia(it, stepx[i], aia_time2iris[closest_time(it, aia_time2iris)])
    interp.append(ob)
    
interp[0].time, interp[0].x, interp[0].aia_time
interp[1].time, interp[1].x, interp[1].aia_time
interp[2].time, interp[2].x, interp[2].aia_time
dx = interp[1].x-interp[0].x
dt = interp[1].time-interp[0].time
dx
dt
interp[0].aia_time-interp[0].time
(interp[0].aia_time-interp[0].time)*dx/dt
interp[0].x+(interp[0].aia_time-interp[0].time)*dx/dt
aia_main[0]
plt.imshow(aia_main[0])
aia_main[1]['NAXIS3']
aia_main[1]['NAXIS1']
plt.imshow(aia_main[0][0,:,:])
aia_main[0][0]
len(aia_time2iris)
interp
interp[0].x
interp[0].time
interp[0].aia_time
t_aia
iris2aia[0]
iris2aia
aia_time2iris
ob = interp[0]
if ob.x > ob.aia_time:
    print("x")
    
if ob.x > aia_time2iris[ob.aia_time]:
    print("x")