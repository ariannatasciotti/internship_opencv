import cv2 as cv
import numpy as np

image = cv.imread("File1Out_noRes_standard.jpg") # input image

lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)   # convert to LAB color space
l, a, b = cv.split(lab)
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # histogram equalization to improve the contrast of the images
cl = clahe.apply(l)
limg = cv.merge((cl,a,b))
final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
final = cv.cvtColor(final, cv.COLOR_BGR2GRAY)

lower = np.array([80])
upper = np.array([130])
mask = cv.inRange(final, lower, upper)
cv.imwrite('mask' + '.png', mask) 


# Simple blob detector

params = cv.SimpleBlobDetector_Params()

# Filter by color
params.filterByColor = True
params.blobColor = 0

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 1000
params.maxArea = 5000000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.01

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1

detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(final)
im_with_keypoints = cv.drawKeypoints(final, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
pts = cv.KeyPoint_convert(keypoints)   # return the blob center coordinates 
cols = pts[:,0]
rows = pts[:,1]

# Here i'm setting all the blob pixels to white in order to do a mask later 

for c in range(len(keypoints)):
    s = round(keypoints[c].size)       # this is the diameter of the blob c
    image[round(rows[c])-round(s/2):round(rows[c])+round(s/2),round(cols[c])-round(s/2):round(cols[c])+round(s/2),0] = 255
    image[round(rows[c])-round(s/2):round(rows[c])+round(s/2),round(cols[c])-round(s/2):round(cols[c])+round(s/2),1] = 255
    image[round(rows[c])-round(s/2):round(rows[c])+round(s/2),round(cols[c])-round(s/2):round(cols[c])+round(s/2),2] = 255
cv.imwrite('Output' + '.png', image) 


# Now try to apply inpaint to the obtained image

img = cv.imread("Output.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
lower = np.array([255])
upper = np.array([255])
mask = cv.inRange(gray, lower, upper)
cv.imwrite('mask' + '.png', mask) 
mask = cv.imread('mask.png',0)
dst = cv.inpaint(image,mask,20,cv.INPAINT_NS)
cv.imwrite('inpainted.jpg', dst)
cv.namedWindow("inpaint", cv.WINDOW_NORMAL)
cv.resizeWindow("inpaint", 1000, 1000)
cv.imshow("inpaint", dst)
cv.waitKey(5000)
cv.destroyAllWindows()
