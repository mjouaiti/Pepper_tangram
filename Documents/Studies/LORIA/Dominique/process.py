import cv2
import numpy as np
from skimage.measure import compare_ssim
import imutils
import time

def process(img):
    img_p = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(img, 50, 150,apertureSize = 3)
    edges = edges[10:-10, 3:-3]


    ret,thresh = cv2.threshold(img,127,255,0)
    thresh = thresh

    kernel = np.ones((9,9),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    cv2.drawContours(img_p, contours, -1, (255,0,0), 3)

    area, index = 0, -1
    for k, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        print w * h, w / h
        if w * h < 15000 and w /h < 3 and w * h > area:
            cv2.rectangle(img_p,(x,y),(x+w,y+h),(0,255,255),2)
            area = w * h
            index = k
        
    x,y,w,h = cv2.boundingRect(contours[index])
    cv2.rectangle(img_p,(x,y),(x+w,y+h),(0,255,0),2)
    return img_p, edges, (x,y,w,h)

def rotate(img):
    edges = cv2.Canny(img, 50, 150,apertureSize = 3)
    edges = edges[3:-3, 10:-10]

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    maxL, angle = 0.0, 0.0
    vertical_lines, horizontal_lines = [], []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        if abs(theta - 0.0) < 0.6:
            vertical_lines.append([x1, x2])
            #cv2.line(img_p,(x1,y1),(x2,y2),(0,0,255),2)
        elif abs(theta - np.pi / 2.0) < 0.6:
            horizontal_lines.append([y1, y2])
            #cv2.line(img_p,(x1,y1),(x2,y2),(255,0,255),2)
            if maxL < abs(rho) and theta < 1.9:
                maxL = rho
                angle = theta
        else:
            pass #cv2.line(img_p,(x1,y1),(x2,y2),(0,255,0),2)
        
    if angle > 0.0001:
        angle -= np.pi / 2.0

    print("Angle: ", angle * 180.0 / np.pi)
    rotated = imutils.rotate(img, angle * 180.0 / np.pi)
    return rotated

alpha = 3. # Contrast control (1.0-3.0)
beta = 10 # Brightness control (0-100)

original = cv2.imread("/Users/Melanie/Movies/Moth2/00028158.JPG", 0)
original = imutils.rotate(original, 180)
original = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)
original_p = rotate(original)

original_p, edges_o, bb_o = process(original_p)

cv2.rectangle(original_p,(bb_o[0],bb_o[1]),(bb_o[0] + bb_o[2],bb_o[1] + bb_o[3]),(255,0,255),2)

_,thresh_o = cv2.threshold(original_p,127,255,0) #[bb_o[1]:bb_o[3], bb_o[0]:bb_o[2]]
#
#kernel = np.ones((3,3),np.uint8)
#thresh_o = cv2.morphologyEx(thresh_o, cv2.MORPH_OPEN, kernel)
#thresh_o = cv2.morphologyEx(thresh_o, cv2.MORPH_CLOSE, kernel)

cv2.imshow("original", original_p)
cv2.imshow("original_edges", edges_o)
cv2.imshow("thresholded", thresh_o)
cv2.waitKey()
