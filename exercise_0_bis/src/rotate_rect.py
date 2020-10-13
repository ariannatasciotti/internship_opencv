import cv2 as cv
import numpy as np

image = cv.imread("Artboard.png")
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
lower = np.array([18, 240, 245])
upper = np.array([25, 255, 255])
mask = cv.inRange(hsv, lower, upper) 
mask_img = cv.bitwise_and(image, image, mask=mask)   
mask_img = 255 - mask
cv.imshow("Original", image)
cv.waitKey(5000) 
edges = cv.Canny(mask_img,100,200)
cv.imshow("edges", edges)
cv.waitKey(5000) 
contours, hierarchy = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
areas = [cv.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cont = contours[max_index]
rect = cv.minAreaRect(cont)
box = cv.boxPoints(rect)
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = box.astype("float32")
dst_pts = np.array([[0, height],
                        [0, 0],
                        [width, 0],
                        [width, height]], dtype="float32")
M = cv.getPerspectiveTransform(src_pts, dst_pts)
new_img = cv.warpPerspective(image, M, (width, height))
cv.imwrite("crop_img.jpg", new_img)
cv.imshow("Cropped", new_img)
cv.waitKey(5000) 