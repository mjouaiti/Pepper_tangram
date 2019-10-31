import cv2
import numpy as np
import imutils
import sys
import scipy.signal

def process(img, prev_bb):
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

    area, index, dist = 0, -1, 10000
    px, py, pw, ph = prev_bb
    for k, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if (px - x)**2 +(py - y)**2 < dist and w * h > 1000 and w /h < 3:
            cv2.rectangle(img_p,(x,y),(x+w,y+h),(0,255,0),2)
            dist = (px - x)**2 +(py - y)**2
            index = k
        
    x,y,w,h = cv2.boundingRect(contours[index])
    cv2.rectangle(img_p,(x,y),(x+w,y+h),(0,255,0),2)
    return img_p, edges, (x,y,w,h)
    
def rotate(img, prev_angle):
    edges = cv2.Canny(img, 50, 150,apertureSize = 3)
    edges = edges[3:-3, 10:-10]

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    try:
        lines[0]
    except:
        rotated = imutils.rotate(img, prev_angle)
        return rotated, prev_angle

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
    if abs(angle) < 0.001:
        angle = prev_angle

    print("Angle: ", angle * 180.0 / np.pi)
    rotated = imutils.rotate(img, angle * 180.0 / np.pi)
    return rotated, angle

path = sys.argv[1]

index = 28000
angles = [0.07]
p_coord = (0, 0, -73, 65)

index += 1

prev_bb, prev_angle = (1280/2,780/2,0,0), 0

while 1:
    original = cv2.imread(path + str(index).zfill(8) + ".JPG", 1)
    try:
        alpha = 3. # Contrast control (1.0-3.0)
        beta = 10 # Brightness control (0-100)
        original = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)
    except:
        break

    rotated, prev_angle = rotate(original, prev_angle)
    #rotated = rotated[max(coord[2:]):max(coord[2:]) + 550,260:-410]
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    original_p, _, prev_bb = process(gray, prev_bb)
    
    #cv2.imwrite("/Users/Melanie/Movies/Moth2/" + str(index).zfill(8) + "_r.png", rotated)
    cv2.imshow("original", original)
    cv2.imshow("rotated", original_p)
    #cv2.imshow("edges", edges)
    cv2.waitKey(1)
    index += 1

cv2.destroyAllWindows()
