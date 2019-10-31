
import cv2

template = cv2.imread("/Users/Melanie/Documents/Studies/LORIA/Dominique/body.jpg", 0)
w, h = template.shape[::-1]
#img = cv2.imread("/Users/Melanie/Movies/Moth2/00028000.JPG", 0)
img = cv2.imread("/Users/Melanie/Movies/Moth2/00028147.JPG", 0)

method = cv2.TM_CCORR #TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
res = cv2.matchTemplate(img, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 255, 2)

cv2.imshow("result", img)
cv2.waitKey(0)
