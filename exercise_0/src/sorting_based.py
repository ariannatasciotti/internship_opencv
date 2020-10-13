import cv2 as cv
import numpy as np
from utilities.sorting_contours import sort_contours
from utilities.shapedetectors import ShapeDetector

# color based image segmentation

image = cv.imread("A4_Shapes.png")
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
lower = np.array([20, 150, 100])
upper = np.array([20, 255, 255])
mask = cv.inRange(hsv, lower, upper)  # filtering yellow
mask_img = cv.bitwise_and(image, image, mask=mask)   # bitwise conjunction of the two arrays image and image
mask_img = 255 - mask
cv.imshow("Original", image)
cv.waitKey(5000) 
cv.imshow("Mask", mask_img)
cv.waitKey(5000) 
cv.destroyAllWindows()
# find contours on mask image
contours, hierarchy = cv.findContours(mask_img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
'''
sd = ShapeDetector()
for c in contours:
    M = cv.moments(c)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    shape = sd.detect(c)
    cv.putText(mask_img.copy(), shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 255, 0), 2)
    cv.imshow("Image", mask_img)
    cv.waitKey(7000) 
'''
# sort contours from left to right (there are 5 bounding boxes)
(cnts, boundingBoxes) = sort_contours(contours, method="left-to-right")
x,y,w,h = boundingBoxes[2]           # this is the bounding box for the bigger rectangle without border
new_img = mask_img[y:y+h, x:x+w]
cv.imwrite('Output_sorting' + '.png', new_img) 
cv.imshow("Output", new_img)
cv.waitKey(9000)
