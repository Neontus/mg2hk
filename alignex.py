import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
from scipy import signal
import argparse

#aligning function
def align_images(image, template, numkp):
    # convert to greyscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    blurredImage = blur(imageGray)
    blurredTemplate = blur(templateGray)
    print("blurred images")
    
    sift = cv2.SIFT_create()
    kpsA, descsA = sift.compute(blurredImage, kpsa)
    kpsB, descsB = sift.compute(blurredTemplate, kpsi)
    
    print(type(kpsa[0]))
    
    nkpsa = []
    nkpsi = []
    
    for kp in kpsa:
        nkpsa.append(kp.pt)
    
    for kp in kpsi:
        nkpsi.append(kp.pt)
    
    nkpsa = np.array(nkpsa)
    nkpsi = np.array(nkpsi)
    
    print(nkpsa.shape)
       
    ptsA = np.zeros((numkp, 2), dtype="float")
    ptsB = np.zeros((numkp, 2), dtype="float")
    
    for i in range(numkp):
        ptsA[i] = nkpsa[i].astype(np.float32)
        ptsB[i] = nkpsi[i].astype(np.float32)
        
        
    # ptsA = ptsA.astype(np.float32)
    # ptsB = ptsB.astype(np.float32)
    M = cv2.estimateAffinePartial2D(nkpsa, nkpsi, False)
    #inversetra = cv2.invertAffineTransform(M[0])
    (h, w) = template.shape[:2]
    
    print("image dimensions: ", image.shape)
    
    #polygoned = cv2.polylines(image, [newRectPoints], True, (0,0,255), 3)
    #cv2.imshow("PLEASE WORK", polygoned)
    #cv2.waitKey(0)
    
    aligned = cv2.warpAffine(image, M[0], (w, h))
    
    aligned_color = cv2.warpAffine(color_aia, M[0], (w, h))
    print("RESULT SHAPE: ", aligned_color.shape)
    
    cv2.imwrite('/Users/jkim/Desktop/mg2hk/output/coord_manual.png', aligned_color)
    
    # return the aligned image
    return aligned, aligned_color

#solve affine transformation
def applyAffine(point, affine):
    x, y = point
    newp = [x, y, 1]
    
    transformed = np.matmul(affine, newp)
    transformed = transformed.astype(np.float32)
    return transformed
    
#blur function
def blur(image):
    dst = cv2.blur(image, (25, 25))
    return dst

def evaluate(im1, im2):
    print("IN CORRELATION")
    print(im1.shape, im2.shape)
    print("""correlation score
channel one: """, np.sum(im1[:,:,0].flat == im2[:,:,0].flat) / im1[:,:,0].size, """
channel two: """, np.sum(im1[:,:,1].flat == im2[:,:,1].flat) / im1[:,:,1].size, """
channel three: """, np.sum(im1[:,:,2].flat == im2[:,:,2].flat) / im1[:,:,2].size)
        
def detectClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if params == "iris":
            print(x, ", ", y, " on iris")
            kpsi.append(cv2.KeyPoint(x, y, 1))
            cv2.circle(template, (x, y), 2, (0, 0, 255), 2)
            cv2.imshow("IRIS THRES", template)
        if params == "aia":
            print(x, ", ", y, " on aia")
            kpsa.append(cv2.KeyPoint(x, y, 1))
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            cv2.imshow("AIA THRES", image)
    

# START OF PROGRAM
print("[CHECKPOINT] Program Loaded")
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True,
	help="path to input template image")
args = vars(ap.parse_args())
print("[CHECKPOINT] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

#select the keypoints
kpsi = [] #iris keypoints
kpsa = [] #aia keypoints

print("[CHECKPOINT] select keypoints")
cv2.imshow("IRIS THRES", template)
cv2.setMouseCallback("IRIS THRES", detectClick, "iris")
cv2.imshow("AIA THRES", image)
cv2.setMouseCallback("AIA THRES", detectClick, "aia")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[UPDATE]: how many keypoints - ", len(kpsi), len(kpsa))

#to apply to color version
color_path = '/Users/jkim/Desktop/mg2hk/output/aia_color_coord.png'
color_aia = cv2.imread(color_path)

# align the images
print("[CHECKPOINT] aligning images...")
aligned, color_aligned = align_images(image, template, len(kpsi))

#for evaluating correlation
#iris_color_path = '/Users/jkim/Desktop/mg2hk/output/iris_to_align_color.png'
#color_iris = cv2.imread(iris_color_path)

#correlation
#evaluate(color_aligned, color_iris)

aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)

# our first output visualization of the image alignment will be a
# side-by-side comparison of the output aligned image and the
# template
print("HERE", image.shape, template.shape)
stacked = np.hstack([aligned, template])


overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
# show the two output image alignment visualizations

cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


