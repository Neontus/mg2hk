import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from iris_lmsalpy import saveall as sv
from scipy.optimize import minimize
import pick_from_LMSAL

#params
#aia, iris, DEBUG, maxFeatures, num_max_points

# to-define
# detect keypoints, show matches
# calculate transformation
# apply transformation and display, apply to color_version
# return aligned image, matrix, width + height

#updated align function
def align(aia, iris, debug = False, num_max_points=3, blurFilter = 3):
    
    blurredaia = lee_filter(aia, blurFilter)
    blurrediris = lee_filter(iris, blurFilter)
    
    fig, ax = plt.subplots(1, 2, figsize=[10,10])
    ax[0].imshow(aia)
    ax[1].imshow(iris)
    plt.show()
    print("SIZE AFTER BLUR", aia.shape, iris.shape)
    
    
    # sift = cv2.SIFT_create()
    # kpsA, descsA = sift.detectAndCompute(aia, None)
    # kpsB, descsB = sift.detectAndCompute(iris, None)
    
    orb = cv2.ORB_create()
    kpsA = orb.detect(aia, None)
    kpsB = orb.detect(iris, None)
    
    kpsA, descsA = orb.compute(aia, kpsA)
    kpsB, descsB = orb.compute(iris, kpsB)
    
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descsA,descsB,k=2)
    
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    good = sorted(good, key = lambda x: x[0].distance)[:num_max_points]
    matches = sorted(matches, key = lambda x: x[0].distance)[:num_max_points]
    
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(good):
        ptsA[i] = kpsA[m[0].queryIdx].pt
        ptsB[i] = kpsB[m[0].trainIdx].pt
    
        
    ptsA = ptsA.astype(np.float32)
    ptsB = ptsB.astype(np.float32)
    M = cv2.estimateAffinePartial2D(ptsA[:num_max_points], ptsB[:num_max_points], False)
    #print(M)
    #print("ESTIMATE: ", M[0].shape)
    inversetra = cv2.invertAffineTransform(M[0])
    (h, w) = iris.shape[:2]
    
    
    
    if debug:
        # matching keypoints
        matchedVis = cv2.drawMatchesKnn(aia, kpsA, iris, kpsB,
            matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)
    
    print("aia dimensions: ", aia.shape)
    print("done aligning")
    
    return M[0], w, h


#aligning function
def align_images(color_aia, aia, iris, outpath, maxFeatures=500, debug = False, 
                num_max_points=3, blurFilter = 3):
    
    # convert to greyscale
    aiaGray = cv2.cvtColor(aia, cv2.COLOR_BGR2GRAY)
    irisGray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
    
    blurredaia = blur(aiaGray, blurFilter)
    blurrediris = blur(irisGray, blurFilter)
    print("blurred images")
    
#     blurredaia = aiaGray.copy()
#     blurrediris = irisGray.copy()
    
    fig, ax = plt.subplots(1, 2, figsize=[10,10])
    ax[0].imshow(blurredaia)
    ax[1].imshow(blurrediris)
    plt.show()
    print("SIZE AFTER BLUR", blurredaia.shape, blurrediris.shape)
    
    
    sift = cv2.SIFT_create()
    kpsA, descsA = sift.detectAndCompute(blurredaia, None)
    kpsB, descsB = sift.detectAndCompute(blurrediris, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descsA,descsB,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    good = sorted(good, key = lambda x: x[0].distance)[:num_max_points]
    matches = sorted(matches, key = lambda x: x[0].distance)[:num_max_points]
    
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(good):
        ptsA[i] = kpsA[m[0].queryIdx].pt
        ptsB[i] = kpsB[m[0].trainIdx].pt
    
        
    ptsA = ptsA.astype(np.float32)
    ptsB = ptsB.astype(np.float32)
    M = cv2.estimateAffinePartial2D(ptsA[:num_max_points], ptsB[:num_max_points], False)
    #print(M)
    #print("ESTIMATE: ", M[0].shape)
    inversetra = cv2.invertAffineTransform(M[0])
    (h, w) = iris.shape[:2]
    
    
    
    if debug:
        #bounding box
        a, b, c, d = (0,0), (w, 0), (w,h), (0, h)
        start, end = a, c
        # boxed = cv2.rectangle(aia, start, end, (0,0,255) , 2)
        # cv2.imshow("bounding box", boxed)
        # cv2.waitKey(0)
        
        #matching keypoints
        matchedVis = cv2.drawMatchesKnn(blurredaia, kpsA, blurrediris, kpsB,
            matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    
        #adjusted box
        #print("inverse matrix: ", inversetra)

        newa, newb, newc, newd = applyAffine(a, inversetra), applyAffine(b, inversetra), applyAffine(c, inversetra), applyAffine(d, inversetra)

        #print("original points: ", a, b, c, d)
        #print("new points: ", newa, newb, newc, newd)
        newRectPoints = np.array([newa, newb, newc, newd], np.int32)
        newRectPoints = newRectPoints.reshape((-1, 1, 2))
    
    print("aia dimensions: ", aia.shape)
    
    polygoned = cv2.polylines(aia, [newRectPoints], True, (0,0,255), 3)
    cv2.imshow("PLEASE WORK", polygoned)
    cv2.waitKey(0)
    
    aligned = cv2.warpAffine(aia, M[0], (w, h))
    
    aligned_color = cv2.warpAffine(color_aia, M[0], (w, h))
    sv.save('quepasa.jbl.gz', aligned_color, M, w, h)
    print("RESULT SHAPE: ", aligned_color.shape)
    
    cv2.imwrite(outpath, aligned_color)
    
    # return the aligned aia
    return aligned, M[0], w, h

#solve affine transformation
def applyAffine(point, affine):
    x, y = point
    newp = [x, y, 1]
    
    transformed = np.matmul(affine, newp)
    transformed = transformed.astype(np.float32)
    return transformed
    
#gaussian blur function
def blur(image, blur_filter):
    dst = cv2.GaussianBlur(image, (blur_filter, blur_filter), 0, 0)
    return dst

#filtering with two thresholds
def imgthr(image, lt, ut):
    fl = image.copy().flatten()
    for i, val in enumerate(fl):
        if np.logical_and((val>lt), (val<ut)):
            fl[i] = True
        else:
            fl[i] = False
    
    return np.reshape(fl, image.shape)


def conservative_smoothing_gray(data, filter_size):
    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i-indexer, i+indexer+1):
                for m in range(j-indexer, j+indexer+1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k,m])
            temp.remove(data[i,j])
            max_value = max(temp)
            min_value = min(temp)
            if data[i,j] > max_value:
                new_image[i,j] = max_value
            elif data[i,j] < min_value:
                new_image[i,j] = min_value
            temp =[]
    return new_image.copy()

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def get_top_n(img, n):
    a = img.copy()
    img_flat = a.flatten()
    sort_flat = np.sort(img_flat)
    l = int(len(sort_flat)*(1-n))
    thr = sort_flat[l]
    
    return thr

def manual_align(aiao, iriso):
    aia = cv2.cvtColor(aiao, cv2.COLOR_GRAY2BGR)
    iris = cv2.cvtColor(iriso, cv2.COLOR_GRAY2BGR)
    
    def detectClick(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if params == "iris":
                print(x, ", ", y, " on iris")
                kpsi.append(cv2.KeyPoint(x, y, 1))
                cv2.circle(iris, (x, y), 2, (0, 0, 255), 2)
                cv2.imshow("IRIS keypoint select", iris)
            if params == "aia":
                print(x, ", ", y, " on aia")
                kpsa.append(cv2.KeyPoint(x, y, 1))
                cv2.circle(aia, (x, y), 2, (255, 0, 0), 2)
                cv2.imshow("AIA keypoint select", aia)

    kpsi = [] #iris keypoints
    kpsa = [] #aia keypoints
    nkpsa = []
    nkpsi = []
    

    print("[CHECKPOINT] select keypoints")
    cv2.imshow("IRIS keypoint select", iris)
    cv2.setMouseCallback("IRIS keypoint select", detectClick, "iris")
    cv2.imshow("AIA keypoint select", aia)
    cv2.setMouseCallback("AIA keypoint select", detectClick, "aia")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[UPDATE]: how many keypoints - ", len(kpsi), len(kpsa))
    
    for kp in kpsa:
        nkpsa.append(kp.pt)
    
    for kp in kpsi:
        nkpsi.append(kp.pt)
    
    nkpsa = np.array(nkpsa)
    nkpsi = np.array(nkpsi)
       
    numkp = len(kpsi)
    
    ptsA = np.zeros((numkp, 2), dtype="float")
    ptsB = np.zeros((numkp, 2), dtype="float")
    
    for i in range(numkp):
        ptsA[i] = nkpsa[i].astype(np.float32)
        ptsB[i] = nkpsi[i].astype(np.float32)
    M = cv2.estimateAffinePartial2D(nkpsa, nkpsi, False)
    (h, w) = iris.shape[:2]

    return M[0], w, h

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

import skimage
from skimage.feature import ORB, match_descriptors, SIFT, plot_matches
from skimage import transform

def sift_ransac(aia, iris, debug = False):
    # print(aia.shape, iris.shape)
    sift = SIFT()
    sift.detect_and_extract(aia)
    kpsa, descsa = sift.keypoints, sift.descriptors
    sift.detect_and_extract(iris)
    kpsi, descsi = sift.keypoints, sift.descriptors

    matches = match_descriptors(descsa, descsi, max_ratio = 0.6, cross_check=True)

    if debug:
        fig, ax = plt.subplots(1, 1, figsize=[5, 5])
        plot_matches(ax, aia, iris, kpsa, kpsi, matches)

        plt.show()

    if len(matches) == 0:
        print("NO MATCHES!!")

    matchcoords = [(kpsa[m[0]], kpsi[m[1]]) for m in matches]
    aiacoords = np.array([np.array([m[0][1], m[0][0]]) for m in matchcoords])
    iriscoords = np.array([np.array([m[1][1], m[1][0]]) for m in matchcoords])

    H, inliers = cv2.estimateAffinePartial2D(aiacoords, iriscoords)

    return H, iris.shape[1], iris.shape[0]

from skimage.feature import CENSURE

def censure_ransac(aia, iris, debug = False):
    print(aia.shape, iris.shape)
    censure = CENSURE()
    censure.detect_and_extract

    sift.detect_and_extract(aia)
    kpsa, descsa = sift.keypoints, sift.descriptors
    sift.detect_and_extract(iris)
    kpsi, descsi = sift.keypoints, sift.descriptors

    matches = match_descriptors(descsa, descsi, max_ratio = 0.6, cross_check=True)

    if debug:
        fig, ax = plt.subplots(1, 1, figsize=[5, 5])
        plot_matches(ax, aia, iris, kpsa, kpsi, matches)
        plt.show()

    matchcoords = [(kpsa[m[0]], kpsi[m[1]]) for m in matches]
    aiacoords = np.array([np.array([m[0][1], m[0][0]]) for m in matchcoords])
    iriscoords = np.array([np.array([m[1][1], m[1][0]]) for m in matchcoords])

    H, inliers = cv2.estimateAffinePartial2D(aiacoords, iriscoords)

    return H, iris.shape[1], iris.shape[0]

class super_align():
    def __init__(self, aia, iris, guess_aia_N, guess_iris_N, guess_blur):
        self.aia = aia
        self.iris = iris
        self.guess_aia_N = guess_aia_N
        self.guess_iris_N = guess_iris_N
        self.guess_blur = guess_blur

    def do_alignment(self, params):
        aia_N = params[0]
        iris_N = params[1]
        blur = params[2]

        # print("HERE: ", aia_N, iris_N, blur)

        #preprocess aia + iris
        AIA_THRESH = get_top_n(self.aia, aia_N)
        IRIS_THRESH_L = get_top_n(self.iris, iris_N)
        IRIS_THRESH_H = 450.025

        aia_to_align = ((self.aia > AIA_THRESH) * 255).astype(np.uint8)
        iris_to_align = cv2.normalize(lee_filter((imgthr(self.iris, IRIS_THRESH_L, IRIS_THRESH_H) * 255), blur), None, 0,255, cv2.NORM_MINMAX).astype('uint8')

        matrix, walign, halign = sift_ransac(aia_to_align, iris_to_align, debug=False)

        # print("check: ", matrix, walign, halign)

        if matrix is not None:
            aligned_color_aia = cv2.warpAffine(self.aia, matrix, (walign, halign))
            error = mse(aligned_color_aia, self.iris)
        else:
            error = 1000000000000000
            exit()

        return error

    def nm_minimize(self):
        res = minimize(self.do_alignment, x0=[self.guess_aia_N, self.guess_iris_N, self.guess_blur], method='Nelder-Mead', tol=1e-2)
        print("RESULT: ", res)
        return res

    def SLSQP_minimize(self):
        bnds = ((0, None), (0, None), (0, None))
        cons = ()
        res = minimize(self.do_alignment, x0=[self.guess_aia_N, self.guess_iris_N, self.guess_blur], method='SLSQP', bounds = bnds, constraints = cons)
        print("RESULT: ", res)
        return res

def load(obsid):
    print("testing with: (OBSID - {})".format(obsid))
    obsid, numraster = OBSID, 0
    iris_file = pick_from_LMSAL.obsid_raster(obsid, raster=numraster)
    aia_file = pick_from_LMSAL.obsid_raster2aia(obsid, raster=numraster, pattern='1600')

    aia_data = my_fits.read(aia_file[0])
    aia_1600 = aia_data[0]
    hdr_aia_1600 = aia_data[1]
    aia_middle_step = int(aia_1600.shape[0] // 2)

    aia_1600 = aia_1600[aia_middle_step, :, :]
    info_1600 = my_fits.read(aia_file[0], ext=1)
    xcen_aia = info_1600[0][:, 10]
    ycen_aia = info_1600[0][:, 11]

    hdr_iris_data = ei.only_header(iris_file[0])
    aux_hdr_iris_data = ei.only_header(iris_file[0], extension=1)
    xcen_iris = hdr_iris_data['XCEN']
    ycen_iris = hdr_iris_data['YCEN']
    xscl_iris = aux_hdr_iris_data['CDELT3']
    yscl_iris = aux_hdr_iris_data['CDELT2']
    xscl_aia = hdr_aia_1600['CDELT1']
    yscl_aia = hdr_aia_1600['CDELT2']

    # Mask + Crop IRIS
    print("-" * 10, "[Section] IRIS Masking + Cropping + WL", "-" * 10)
    iris_raster = ei.load(iris_file[0])
    extent_hx_hy = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
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
    print(extent_iris)
    # extent_iris = iris_raster.raster['Mg II k 2796'].extent_heliox_helioy
    # print(extent_iris)

    # Scale AIA to arcsec
    print("-" * 10, "[Section] Reshaping AIA to IRIS", "-" * 10)
    try_aia = aia_1600.shape[0] * yscl_aia, aia_1600.shape[1] * xscl_aia
    new_aia = rebin.congrid(aia_1600, try_aia)
    haia, waia = new_aia.shape
    print('AIA size', haia, waia)
    extent_aia = [xcen_aia[aia_middle_step] - waia / 2, xcen_aia[aia_middle_step] + waia / 2,
                  ycen_aia[aia_middle_step] - haia / 2, ycen_aia[aia_middle_step] + haia / 2]

    # Cropping AIA to IRIS
    print("-" * 10, "[Section] Cropping AIA to IRIS", "-" * 10)
    pad = 0
    acp = [(extent_iris[0] - pad, extent_iris[3] + pad), (extent_iris[0] - pad, extent_iris[2] - pad),
           (extent_iris[1] + pad, extent_iris[3] + pad), (extent_iris[1] + pad, extent_iris[2] - pad)]
    x_i = int(extent_iris[0] - pad - extent_aia[0])
    x_f = int(extent_iris[1] + pad - extent_aia[0])
    y_f = -int(extent_iris[3] + pad - extent_aia[3])
    y_i = -int(extent_iris[2] - pad - extent_aia[3])
    cut_aia = new_aia[y_f:y_i, x_i:x_f]

    return new_iris_data, cut_aia

#visualizing
import numpy as np
import argparse
import imutils
import cv2