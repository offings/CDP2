import cv2
from numpy import percentile
import imutils
import matplotlib.pyplot as plt
import numpy as np
from numpy import percentile

img = cv2.imread('image/second_output/28.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

lower = percentile(gray, 0)
higher = percentile(gray, 100)

cv2.normalize(gray, gray, lower-100, higher+100, cv2.NORM_MINMAX)

height, width = gray.shape
print(height, width)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

CANNY_THRESH_1 = 40
CANNY_THRESH_2 = 100

edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

contour_info = []
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
for c in contours:
    contour_info.append([
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ])
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

for area in range(len(contour_info)) :
    print(contour_info[area][2])

cv2.drawContours(img, contour_info[0], 0, (0, 0, 255), 5)

print("\nresult")
# for i in range(len(contour_info[0][0])) :
#     x, y = contour_info[0][0][i][0]
#     print(x, y)

#
# # src = imutils.resize(img, width=800)
cv2.imwrite("contour.jpg", img)
# cv2.imshow("src", img)
plt.figure()
plt.imshow(img, cmap = "gray")
plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()