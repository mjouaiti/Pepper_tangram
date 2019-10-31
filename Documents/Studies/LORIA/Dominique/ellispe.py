import numpy as np
import cv2

img_object = cv2.imread("/Users/Melanie/Documents/Studies/LORIA/Dominique/body.jpg", 0)
img_object = cv2.Canny(img_object, 50, 150,apertureSize = 3)
#img = cv2.imread("/Users/Melanie/Movies/Moth2/00028000.JPG", 0)
img_scene = cv2.imread("/Users/Melanie/Movies/Moth2/00028147.JPG", 0)[120:-50,420:-300]
edges = cv2.Canny(img_scene, 50, 150,apertureSize = 3)

_,thresh = cv2.threshold(img_scene,127,255,0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

minEllipse = [None]*len(contours)
for i, c in enumerate(contours):
    if c.shape[0] > 5:
        minEllipse[i] = cv2.fitEllipse(c)


cv2.drawContours(img_scene, contours, -1, (0, 0, 255))

for i, c in enumerate(contours):
    if c.shape[0] > 5:
        cv2.ellipse(img_scene, minEllipse[i], (255, 0, 0), 2)
        
cv2.imshow("ellipse", img_scene)
cv2.waitKey(0)
