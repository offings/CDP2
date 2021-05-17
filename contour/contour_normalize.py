import cv2
from numpy import percentile
import imutils
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('image/first_output/33.jpg')
out = cv2.imread('image/second_output/33.jpg')
out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
output = np.array(out)

height, width = output.shape
print(height, width)

# lower = percentile(img, 0)
# higher = percentile(img, 100)
# cv2.normalize(img, img, lower-100, higher+500, cv2.NORM_MINMAX)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

CANNY_THRESH_1 = 10
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

print("result")

for area in range(len(contour_info)) :
    if(contour_info[area][2] <= 1100.0 and contour_info[area][2] >= 200.0) :
        print(area, contour_info[area][2])
        cv2.drawContours(img, contour_info[area], 0, (0, 0, 255), 5)

# for i in range(len(contours)) :
#     cv2.drawContours(img, [contours[i]], 0, (0, 0, 255), 5)

# # contours[hierarchy][duplcated_list][point]
# sum1 = 0
# sum2 = 0
#
# print("first")
# for i in range(len(contours[1])) :
#     x,y = contours[1][i][0]
#     sum1 = sum1 + output[x][y]
#     print(x,y,output[x][y])
#
# print("second")
# for j in range(len(contours[2])) :
#     x,y = contours[2][j][0]
#     sum2 = sum2 + output[x][y]
#     print(x, y, output[x][y])
#
# ave1 = sum1 / len(contours[1])
# ave2 = sum2 / len(contours[2])
#
# print(ave1, ave2)
#
# # src = imutils.resize(img, width=800)
cv2.imwrite("contour.jpg", img)
# cv2.imshow("src", img)
# plt.figure()
# plt.imshow(output, cmap = "gray")
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()