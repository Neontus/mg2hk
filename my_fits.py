import numpy as np
import os
from astropy.io import fits


def write(name, data, header=None, force=False):

    hdu = fits.PrimaryHDU(data)
    hdr = hdu.header
    if header != None:
        if isinstance(header, fits.header.Header):
            print('Dentro')
            hdr = header
        else:
            for i in header.keys():
                hdr[i] = header[i]
                if isinstance(header[i], str):
                    hdr[i] = header[i].replace('\n', '')
        hdu = fits.PrimaryHDU(data, header=hdr)

    if os.path.isfile(name) == True and force == True: os.system('rm {}'.format(name))

    hdu.writeto(name)
  
    return
     
def read(filename, verbose=True, ext=0):

    hdulist = fits.open(filename)
    if verbose !=0:
       print()
       print('Reading file {}... '.format(filename))
       hdulist.info()
    hdr = hdulist[ext].header

    data = hdulist[ext].data

    hdulist.close()

    return data, hdr


#a = np.zeros((10,30))
#b = np.random.random(100)
#c = np.array((3,3,3))
#my_fits.write_list('pru.fits', [a,b,c], force=True, name_var=['a', 'b', 'c'])

def write_list(name, ori_list_data, header=None, force=False, name_var = None, info_var=None, transpose=True):


    list_data = []
    for l in ori_list_data:
        if transpose:
            list_data.append(np.transpose(l))
        else:
            list_data.append(l)

    hdr = fits.Header()
    if name_var != None: hdr['NAME_VAR'] = name_var[0]
    if info_var != None: hdr['INFO_VAR'] = info_var[0]
    hdu = fits.PrimaryHDU(list_data[0], header=hdr)

    hdul = fits.HDUList(hdu)

    hdr = hdu.header
    if header != None:
        if isinstance(header, fits.header.Header):
            hdr = header
        else:
            for i in header.keys():
                hdr[i] = header[i]
    if len(list_data) > 1:
       for j, i in enumerate(list_data[1:]): 
           #hdrl = False 
           hdrl = fits.Header()
           if name_var != None:
               hdrl['NAME_VAR'] = name_var[j+1]
           if info_var != None:
               hdrl['INFO_VAR'] = info_var[j+1]
           hdul.append(fits.ImageHDU(i, header=hdrl))


    if os.path.isfile(name) == True and force == True: os.system('rm {}'.format(name))

    hdul.writeto(name)

    return
