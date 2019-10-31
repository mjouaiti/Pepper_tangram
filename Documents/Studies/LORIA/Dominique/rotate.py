import cv2
import numpy as np
import imutils

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
    
def contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


#original = cv2.imread("/Users/Melanie/Downloads/test.jpg", 1)#
original = cv2.imread("/Users/Melanie/Movies/Moth2/00028279.JPG", 1)
original = imutils.rotate(original, 180)
alpha = 3. # Contrast control (1.0-3.0)
beta = 10 # Brightness control (0-100)
original = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)
alpha = 1. # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
original = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
############################### Rotation
edges = cv2.Canny(gray, 50, 150,apertureSize = 3)
edges = edges[3:-3, 10:-10]

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

angle, maxL = 0.0, 0.0
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
        cv2.line(original,(x1,y1),(x2,y2),(0,0,255),2)
    elif abs(theta - np.pi / 2.0) < 0.6:
        horizontal_lines.append([y1, y2])
        cv2.line(original,(x1,y1),(x2,y2),(255,0,255),2)
        print x1, x2, y1, y2
        if maxL < abs(rho) and theta < 1.9:
            maxL = rho
            angle = theta
    else:
        cv2.line(original,(x1,y1),(x2,y2),(0,255,0),2)

    print theta * 180.0 / np.pi, rho, y1, y2

if angle > 0.0001:
    angle -= np.pi / 2.0
#
rotated = imutils.rotate(original, angle * 180.0 / np.pi)
horizontal_lines = np.array(horizontal_lines).flatten()
if len(horizontal_lines) == 0:
    horizontal_lines = np.append([0], horizontal_lines)
    horizontal_lines.append([720], horizontal_lines)
if len(horizontal_lines[horizontal_lines < 720 / 2]) == 0:
    horizontal_lines = np.append(horizontal_lines, [np.max(horizontal_lines[horizontal_lines > 720 / 2]) - 600])
vertical_lines = np.array(vertical_lines).flatten()
if len(vertical_lines) == 0:
    vertical_lines = np.append([410], vertical_lines)
    vertical_lines = np.append([1010], vertical_lines)
print horizontal_lines, vertical_lines, np.min(horizontal_lines[horizontal_lines < 720 / 2])
rotated = rotated[np.min(horizontal_lines[horizontal_lines < 720 / 2]):np.max(horizontal_lines[horizontal_lines > 720 / 2])-30, int(np.max(vertical_lines[vertical_lines < 1280 / 2])) :int(np.max(vertical_lines[vertical_lines > 1280 / 2]))]

#################################### Processing

gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150,apertureSize = 3)
edges = edges[10:-10, 3:-3]


ret,thresh = cv2.threshold(gray,127,255,0)

kernel = np.ones((9,9),np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)#cv2.dilate(thresh,kernel,iterations = 1)
#thresh = cv2.erode(thresh,kernel,iterations = 2)
#thresh = cv2.dilate(thresh,kernel,iterations = 2)
#thresh = cv2.erode(thresh,kernel,iterations = 1)


cv2.imshow("bw", thresh)

contours, hierarchy = cv2.findContours(thresh, 1, 2)
cv2.drawContours(rotated, contours, -1, (255,0,0), 3)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(rotated,(x,y),(x+w,y+h),(0,255,0),2)
#
cv2.imshow("edges", edges)
cv2.imshow("rotated", rotated)
cv2.imshow("original", original)
cv2.waitKey(0)

cv2.destroyAllWindows()
