import numpy as np
import cv2
import math
from scipy import ndimage
from numpy import percentile

img_before = cv2.imread('image/final_image/15.jpg')

# cv2.namedWindow("Before", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Before", 500, 500)
# cv2.moveWindow("Before", 50, 50)
# cv2.imshow("Before", img_before)
# key = cv2.waitKey(0)

img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)

lower = percentile(img_gray, 0)
higher = percentile(img_gray, 100)

cv2.normalize(img_gray, img_gray, lower-100, higher+100, cv2.NORM_MINMAX)

img_edges = cv2.Canny(img_gray, 10, 100)
# HoughLines를 통해 이미지의 선 감지
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=70, maxLineGap=5)

angles = []

for x1, y1, x2, y2 in lines[0]:
    cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

median_angle = np.median(angles)

print("Angle is {}".format(median_angle))