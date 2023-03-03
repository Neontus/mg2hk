# Author: Alberto Sainz Dalda <asainz.solarphysics@gmail.com>

""" Routines to get the filename and path or both for an
    IRIS Level 2 data stored at LMSAL """



from iris_lmsalpy import find

def get(filename):
    
    pos = filename.find('iris_l2')
    date = filename[pos+8:pos+8+15]
    yy = date[0:4]
    mm = date[4:6]
    dd = date[6:8]

    dir_at_lmsal ='/irisa/data/level2/{}/{}/{}/'.format(yy,mm,dd)
    #print('/irisa/data/level2/{}/{}/{}/'.format(yy,mm,dd))

    out = find.find(dir_at_lmsal, filename+'*')
    #out = -1
    #if find.path.isfile(file[0]): out = file
    full_dir = ''
    if len(out) > 0: full_dir = out[0][0:out[0].find('iris_l2')]

    return out, full_dir


def obsid_raster(obsid, raster=0):

    pos = 0
    date = obsid[:8]
    yy = date[0:4]
    mm = date[4:6]
    dd = date[6:8]

    dir_at_lmsal ='/irisa/data/level2/{}/{}/{}/{}/'.format(yy,mm,dd,obsid)
    #print('/irisa/data/level2/{}/{}/{}/'.format(yy,mm,dd))

    out = find.find(dir_at_lmsal, '*{0}*raster*_r{1:05d}.fits'.format(obsid, raster))

    return out


def obsid_raster2aia(obsid, raster=0, pattern=''):

    pos = 0
    date = obsid[:8]
    yy = date[0:4]
    mm = date[4:6]
    dd = date[6:8]

    dir_at_lmsal ='/sedna/ssw/irisa/data/level2/{}/{}/{}/{}/aia/'.format(yy,mm,dd,obsid)
    #print('/irisa/data/level2/{}/{}/{}/'.format(yy,mm,dd))
    print(dir_at_lmsal)	

    out = find.find(dir_at_lmsal, 'aia*{0}*{1}*.fits'.format(obsid, pattern))

    return out






