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
# print(height, width)

# lower = percentile(img, 0)
# higher = percentile(img, 100)
# cv2.normalize(img, img, lower-100, higher+500, cv2.NORM_MINMAX)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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

# for area in range(len(contour_info)) :
#     print(contour_info[area][2])
#
# print("result")

# hole_list : [hierarchy min_x min_y]를 저장할 list -> contour의 게층에 해당하는 x좌표와 y좌표를 list로 저장한다.
hole_list = []

# 원하는 area size만큼의 contour들을 draw하고 해당 index와 area size를 출력하는 부분
for area in range(len(contour_info)) :
    if(contour_info[area][2] <= 1100.0 and contour_info[area][2] >= 200.0) :
        # print(str(contour_info[area][2])+"("+str(area)+")")
        hole_list.append([
            area, 1024, 1024
        ])
    if area == 3 or area == 5 :
        cv2.drawContours(img, contour_info[area], 0, (0, 0, 255), 5)

# 전체 contour를 그리는 코드(해당 코드는 contour들이 sorting되어있지 않음)
# for i in range(len(contours)) :
#     cv2.drawContours(img, [contours[i]], 0, (0, 0, 255), 5)

if len(hole_list) > 4 :
    hole_list = hole_list[:4]

for i in range(len(hole_list)) :
    for j in range(len(contour_info[hole_list[i][0]][0])) :
        x,y = contour_info[hole_list[i][0]][0][j][0]
        if x < hole_list[i][1] :
            hole_list[i][1] = x
        if y < hole_list[i][2] :
            hole_list[i][2] = y

result_x = sorted(hole_list, key=lambda h: h[1], reverse=True)
result_y = sorted(hole_list, key=lambda h: h[2], reverse=True)

max_x = result_x[0][1] - result_x[len(result_x)-1][1]
max_y = result_y[0][2] - result_y[len(result_y)-1][2]

result_hierarch = []

if max_x > max_y :
    result_hierarch.append(result_x[0][0])
    result_hierarch.append(result_x[len(result_x)-1][0])
else :
    result_hierarch.append(result_y[0][0])
    result_hierarch.append(result_y[len(result_y)-1][0])

print(result_hierarch)

# contours[hierarchy][duplcated_list][point]

max_value1 = 0
min_value1 = 256
max_value2 = 0
min_value2 = 256

sum1 = 0
sum2 = 0

for i in range(len(contour_info[result_hierarch[0]][0])) :
    x, y = contour_info[result_hierarch[0]][0][i][0]
    if max_value1 < output[y][x] :
        max_value1 = output[y][x]
    if min_value1 > output[y][x] :
        min_value1 = output[y][x]
    sum1 = sum1 + output[y][x]
    # print(x, y, output[y][x])

for j in range(len(contour_info[result_hierarch[1]][0])) :
    x, y = contour_info[result_hierarch[1]][0][j][0]
    if max_value2 < output[y][x] :
        max_value2 = output[y][x]
    if min_value2 > output[y][x]:
        min_value2 = output[y][x]
    sum2 = sum2 + output[y][x]
    # print(x, y, output[y][x])

ave1 = sum1 / len(contour_info[result_hierarch[0]][0])
ave2 = sum2 / len(contour_info[result_hierarch[1]][0])

final_ave1 = (max_value1 + min_value1) / 2
final_ave2 = (max_value2 + min_value2) / 2

print("\nlight max\nfirst :", max_value1, "\nsecond :", max_value2)
print("\nlight min\nfirst :", min_value1, "\nsecond :", min_value2)
print("\naverage\nfirst :", ave1, "\nsecond :", ave2)
print("\nfinal average\nfirst : ", final_ave1, "\nsecond : ", final_ave2)
#
# # src = imutils.resize(img, width=800)
cv2.imwrite("contour.jpg", img)
# cv2.imshow("src", img)
# plt.figure()
# plt.imshow(output, cmap = "gray")
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()