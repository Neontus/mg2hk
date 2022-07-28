import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

#params
#aia, iris, DEBUG, maxFeatures, num_max_points

#aligning function
def align_images(color_aia, aia, iris, outpath, maxFeatures=500, debug = False, 
                num_max_points=3):
    # convert to greyscale
    aiaGray = cv2.cvtColor(aia, cv2.COLOR_BGR2GRAY)
    irisGray = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)
    
    blurredaia = blur(aiaGray)
    blurrediris = blur(irisGray)
    print("blurred images")
    
    sift = cv2.SIFT_create()
    kpsA, descsA = sift.detectAndCompute(blurredaia,None)
    kpsB, descsB = sift.detectAndCompute(blurrediris,None)
    
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
        
        #matching points
        matchedVis = cv2.drawMatchesKnn(blur(aia), kpsA, blur(iris), kpsB,
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
    print("RESULT SHAPE: ", aligned_color.shape)
    
    cv2.imwrite(outpath, aligned_color)
    
    # return the aligned aia
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
    dst = cv2.blur(image, (30, 30))
    return dst


#visualizing
import numpy as np
import argparse
import imutils
import cv2