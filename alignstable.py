import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

color_path = '/Users/jkim/Desktop/mg2hk/output/aia_color_to_align.png'
color_aia = cv2.imread(color_path)

#aligning function
def align_images(image, template, maxFeatures=500, debug = False, 
                num_max_points=3):
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
            matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    
        #adjusted box
        
        print("inverse matrix: ", inversetra)

        newa, newb, newc, newd = applyAffine(a, inversetra), applyAffine(b, inversetra), applyAffine(c, inversetra), applyAffine(d, inversetra)

        print("original points: ", a, b, c, d)
        print("new points: ", newa, newb, newc, newd)
        newRectPoints = np.array([newa, newb, newc, newd], np.int32)
        newRectPoints = newRectPoints.reshape((-1, 1, 2))
    
    print("image dimensions: ", image.shape)
    
    polygoned = cv2.polylines(image, [newRectPoints], True, (0,0,255), 3)
    cv2.imshow("PLEASE WORK", polygoned)
    cv2.waitKey(0)
    
    aligned = cv2.warpAffine(image, M[0], (w, h))
    
    aligned_color = cv2.warpAffine(color_aia, M[0], (w, h))
    print("RESULT SHAPE: ", aligned_color.shape)
    
    cv2.imwrite('/Users/jkim/Desktop/mg2hk/output/cut_aligned_colorf.png', aligned_color)
    
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
    dst = cv2.blur(image, (16, 16))
    return dst



#visualizing
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True,
	help="path to input template image")
args = vars(ap.parse_args())
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
# align the images
print("[INFO] aligning images...")
aligned = align_images(image, template, maxFeatures = 150, debug = True, num_max_points=50)

aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)
# our first output visualization of the image alignment will be a
# side-by-side comparison of the output aligned image and the
# template
stacked = np.hstack([aligned, template])


overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
# show the two output image alignment visualizations
cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)

