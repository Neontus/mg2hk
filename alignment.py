import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

#aligning function
def align_images(image, template, maxFeatures=500, keepPercent=0.2,debug=False):
    # convert to greyscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    blurredImage = blur(imageGray)
    blurredTemplate = blur(templateGray)
    print("blurred images")
    
    # detect keypoints using ORB
    #orb = cv2.ORB_create()
    #(kpsA, descsA) = orb.detectAndCompute(blurredImage, None)
    #(kpsB, descsB) = orb.detectAndCompute(blurredTemplate, None)
    
    #trying SIFT
    sift = cv2.SIFT_create()
    kpsA, descsA = sift.detectAndCompute(blurredImage,None)
    kpsB, descsB = sift.detectAndCompute(blurredTemplate,None)
    
    
    # match the features
    #method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    #matcher = cv2.DescriptorMatcher_create(method)
    #matches = matcher.match(descsA, descsB, None)
    
    #using other matching algos
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descsA,descsB,k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    good = sorted(good, key = lambda x: x[0].distance)[:3]
    
    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    #matches = sorted(matches, key=lambda x:x.distance)
    
    # keep only the top matches
    #keep = int(len(matches) * keepPercent)
    #matches = matches[:keep]
    
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatchesKnn(image, kpsA, template, kpsB,
            good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    
    # loop over the top matches
    for (i, m) in enumerate(good):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m[0].queryIdx].pt
        ptsB[i] = kpsB[m[0].trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    #(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    
    #converting to float32 for affine transformation
    ptsA = ptsA.astype(np.float32)
    ptsB = ptsB.astype(np.float32)
    tra = cv2.getAffineTransform(ptsA[:3], ptsB[:3])
    inversetra = cv2.invertAffineTransform(tra)
    
    
    print(tra)
    
    (h, w) = template.shape[:2]    
    #aligned = cv2.warpPerspective(image, H, (w, h))
    aligned = cv2.warpAffine(image, tra, (w, h))
    
    cv2.imwrite('/Users/jkim/Desktop/mg2hk/output/aligned.png', aligned)
    
    # return the aligned image
    return aligned



#gaussian blur function
def blur(image):
    dst = cv2.blur(image, (20, 20))
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
aligned = align_images(image, template, maxFeatures = 50, keepPercent = 0.2, debug=False)

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