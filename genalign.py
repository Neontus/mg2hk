import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import imutils
import cv2
import rebin

from iris_lmsalpy import extract_irisL2data as ei
from aiapy.calibrate import normalize_exposure, register, update_pointing
from astropy.io import fits

outpath = "/Users/jkim/Desktop/mg2hk/output/"

def saveblank(fname): 
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    plt.savefig(outpath+fname+'.png', bbox_inches = 'tight',
        pad_inches = 0)

def load_iris(iris_path):
    iris_raster = ei.load(iris_path)
    extent_hx_hy = [330.92126959698425, 426.6590563250996, -489.16880000000003, -306.8492]
    
    extent_iris = extent_hx_hy
    wiris = int(extent_iris[1]-extent_iris[0])+20
    hiris = int(extent_iris[3]-extent_iris[2])+20
    
    mgii = iris_raster.raster["Mg II k 2796"].data[:,:,28]
    iris_img = rebin.congrid(mgii, [685, 360])
    fig, ax = plt.subplots(figsize=[5, 10])
    plt.title = "iris_thres_vis"
    ax.imshow(iris_img>68, origin="lower", cmap="afmhot", interpolation=None)
    saveblank('iris_ready')
    return [wiris, hiris]
    
def load_aia(aia_folder, from_iris):
    wiris, hiris = from_iris
    
    aia_paths = [x for x in os.listdir(aia_folder)]
    to_examine = [3]#, 7, 8]
    #only using 1600
    to_examine = [aia_paths[i] for i in to_examine]
    img_data_list = []
    header_data_list = []
    
    for aia in to_examine:
        hdul = fits.open(aia_folder+aia)
        img_data = hdul[0].data
        header_data_list.append(hdul[0].header)
        hdul.close()
        img_data_list.append(img_data)
    
    dim_aia = img_data_list[0].shape    
    xc, yc = dim_aia[2]//2, dim_aia[1]//2
    
    fig, ax = plt.subplots(figsize=[5, 10])
    ax.imshow(img_data_list[0][dim_aia[0]-1,yc-hiris:yc+hiris, xc-wiris:xc+wiris,]>103, origin='lower', cmap = 'afmhot')
    saveblank('aia_ready')

    
#aligning function
def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug = False):
    # convert to greyscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    blurredImage = blur(imageGray)
    blurredTemplate = blur(templateGray)
    print("blurred images")
    
    sift = cv2.SIFT_create()
    kpsA, descsA = sift.detectAndCompute(blurredImage,None)
    kpsB, descsB = sift.detectAndCompute(blurredTemplate,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descsA,descsB,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    good = sorted(good, key = lambda x: x[0].distance)[:3]
    
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(good):
        ptsA[i] = kpsA[m[0].queryIdx].pt
        ptsB[i] = kpsB[m[0].trainIdx].pt
        
        
    ptsA = ptsA.astype(np.float32)
    ptsB = ptsB.astype(np.float32)
    tra = cv2.getAffineTransform(ptsA[:3], ptsB[:3])
    inversetra = cv2.invertAffineTransform(tra)
    (h, w) = template.shape[:2]
    
    
    
    if debug:
        #bounding box
        a, b, c, d = (0,0), (w, 0), (w,h), (0, h)
        start, end = a, c
        boxed = cv2.rectangle(image, start, end, (0,0,255) , 2)
        cv2.imshow("bounding box", boxed)
        cv2.waitKey(0)
        
        #matching points
        matchedVis = cv2.drawMatchesKnn(blur(image), kpsA, blur(template), kpsB,
            good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    
        #adjusted box
        
        #print("inverse matrix: ", inversetra)

        newa, newb, newc, newd = applyAffine(a, inversetra), applyAffine(b, inversetra), applyAffine(c, inversetra), applyAffine(d, inversetra)

        print("original points: ", a, b, c, d)
        print("new points: ", newa, newb, newc, newd)
        newRectPoints = np.array([newa, newb, newc, newd], np.int32)
        newRectPoints = newRectPoints.reshape((-1, 1, 2))
    
    print("image dimensions: ", image.shape)
    
    polygoned = cv2.polylines(image, [newRectPoints], True, (0,0,255), 3)
    cv2.imshow("PLEASE WORK", polygoned)
    cv2.waitKey(0)
    
    aligned = cv2.warpAffine(image, tra, (w, h))
    
    aligned_color = cv2.warpAffine(color_aia, tra, (w, h))
    print("RESULT SHAPE: ", aligned_color.shape)
    
    cv2.imwrite('/Users/jkim/Desktop/mg2hk/output/aligned_colorf.png', aligned_color)
    
    # return the aligned image
    return aligned

#solve affine transformation
def applyAffine(point, affine):
    x, y = point
    newp = [x, y, 1]
    
    transformed = np.matmul(affine, newp)
    transformed = transformed.astype(np.float32)
    return transformed
    
#gaussian blur function
def blur(image):
    dst = cv2.blur(image, (20, 20))
    return dst




ap = argparse.ArgumentParser()
ap.add_argument("-a", "--aiapath", required=True, help = "path to AIA FITS folder")
ap.add_argument("-i", "--irispath", required=True, help = "path to IRIS FITS file")
args = vars(ap.parse_args())
print("Checkpoint: Loading Files")
iris_vars = load_iris(args["irispath"])
load_aia(args["aiapath"], iris_vars)


image = cv2.imread(outpath + "aia_ready.png")
template = cv2.imread(outpath + "iris_ready.png")

# align the images
print("[INFO] aligning images...")
aligned = align_images(image, template, maxFeatures = 50, keepPercent = 0.2, debug = True)

aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)
stacked = np.hstack([aligned, template])


overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
# show the two output image alignment visualizations
cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)