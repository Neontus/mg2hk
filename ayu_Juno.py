import pick_from_LMSAL
import my_fits

obsid, numraster = '20211015_051453_3600009176', 0
iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=numraster)
#aia_file = pick_from_LMSAL.obsid_raster2aia(obsid, raster=numraster)
aia_file = pick_from_LMSAL.obsid_raster2aia(obsid, raster=numraster, pattern='1600')
#k
aia_data = my_fits.read(aia_file[0])
aia_1600 = aia_data[0]
hdr_aia_1600 = aia_data[1]
aia_middle_step = int(aia_1600.shape[0]//2)
aia_middle_step = ei.get_ori_val(xcen_aia, a.raster['Mg II k 2796'].XCEN)
aia_1600 = aia_1600[aia_middle_step,:,:]
info_1600 = my_fits.read(aia_file[0], ext=1)
xcen_aia= info_1600[0][:,10]
ycen_aia = info_1600[0][:,11]
hdr_iris_data = ei.only_header(iris_file[0])
aux_hdr_iris_data = ei.only_header(iris_file[0], extension=1)
xcen_iris = hdr_iris_data['XCEN']
ycen_iris = hdr_iris_data['YCEN']
xscl_iris = aux_hdr_iris_data['CDELT3']
yscl_iris = aux_hdr_iris_data['CDELT2']
xscl_aia = hdr_aia_1600['CDELT1']
yscl_aia = hdr_aia_1600['CDELT2']
#
print()
print('IRIS coordinates: [{},{}]'.format(xcen_iris, ycen_iris))
print('AIA coordinates (closest ima. #{}): [{},{}]'.format(aia_middle_step, xcen_aia[aia_middle_step], ycen_aia[aia_middle_step]))
print()