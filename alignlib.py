import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from iris_lmsalpy import saveall as sv

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
    # lower is better
    return err

#visualizing
import numpy as np
import argparse
import imutils
import cv2