import cv2
from numpy import percentile
import imutils
import matplotlib.pyplot as plt
import numpy as np

def detect_screw():
    img = cv2.imread('image/second_output/25.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape
    # print(height, width)

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

    # for area in range(len(contour_info)):
    #     print(contour_info[area][2])

    cv2.drawContours(img, contour_info[0], 0, (0, 0, 255), 5)

    x_screw = []
    y_screw = []
    # print("\nresult")
    for i in range(len(contour_info[0][0])):
        x, y = contour_info[0][0][i][0]
        x_screw.append(x)
        y_screw.append(y)
        zip_list = zip(x_screw, y_screw)
    # print(list(zip_list))

    screw = list(zip_list).copy()
    # print(screw)

    #
    # # src = imutils.resize(img, width=800)
    cv2.imwrite("contour.jpg", img)
    # cv2.imshow("src", img)
    # plt.figure()
    # plt.imshow(img, cmap = "gray")
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return screw

def multithreshold(img, x, y):
    thresholds = []
    thresholds.append(x)
    thresholds.append(y)
    print(thresholds)

    masks = np.zeros((len(thresholds) + 1, img.shape[0], img.shape[1]), bool)
    for i, t in enumerate(sorted(thresholds)):
        masks[i+1] = (img > t)
    masks[0] = ~masks[1]
    for i in range(1, len(masks) - 1):
        masks[i] = masks[i] ^ masks[i+1]
    return masks

def show_thresholds(src_img, dst_img):
    colors=[0]
    masks = multithreshold(src_img, x, y)
    mask_len=len(masks)
    mask_len=mask_len-1

    for i in range(1, mask_len+1) :
        c_len=int(255/mask_len*(i))
        colors.append(c_len)

    for i, mask in enumerate(masks):
        dst_img[mask]=colors[i]

    return dst_img

# def main(path='./second_output/7.jpg'):

#
# if __name__ == '__main__':
#     main()

img = cv2.imread('image/first_output/25.jpg')
out = cv2.imread('image/second_output/31.jpg')
out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
output = np.array(out)

height, width = output.shape

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

re_area = 0
x_list = []
y_list = []

# 원하는 area size만큼의 contour들을 draw하고 해당 index와 area size를 출력하는 부분
for area in range(len(contour_info)) :
    if(contour_info[area][2] <= 1100.0 and contour_info[area][2] >= 200.0) :
       # print(str(contour_info[area][2])+"("+str(area)+")")
       # print(contour_info[area][0])
        cv2.drawContours(img, contour_info[area], 0, (0, 0, 255), 5)
        re_area+=1
        if (re_area==2):
            x_origin, y_origin = contour_info[area][0][0][0]
            # print(x_origin, y_origin)
            for i in range(len(contour_info[area][0])):
                x,y=contour_info[area][0][i][0]
                x_list.append(x)
                y_list.append(y)
                if (x==x_origin):
                    y_max = max(y_list)
                center_x = int((min(x_list) + max(x_list))/2)
                center_y = int((y_origin + y_max)/2)
                cir_x = center_x - 18
            # print(max(x_list), min(x_list), y_max)
            # print(center_x, center_y)
            break

screw_result = []
screw_result = detect_screw()

for i in range(len(screw_result)):
    if(screw_result[i][0] == x_origin and screw_result[i][1] > y_max):
        x_line = x_origin
        y_line = screw_result[i][1]
y_mid = int((y_line + y_max)/2)

cv2.imwrite("contour.jpg", img)

L = 256  # number of levels
path='image/second_output/25.jpg'
img2 = cv2.imread(path, 0)  # read image in as grayscale

# print(img.shape) # print pixel size

min = np.amin(img2)
max = np.amax(img2)
print(min, max)
scaled_img = img2.copy()
for i in range(0, img2.shape[0]):
    for j in range(0, img2.shape[1]):
        scaled_img[i][j] = (img2[i][j] - min) / (max - min) * 255

temp1 = scaled_img[center_y][center_x]
temp2 = scaled_img[center_y][cir_x]
temp3 = scaled_img[y_mid][x_origin]
print(temp1, temp2, temp3)
print(center_y, center_x, cir_x, y_mid, x_origin)

x = int((temp1 + temp3)/2)
y = int((temp2 + temp3)/2)

# Calculate histogram
hist = cv2.calcHist(
    [scaled_img],
    channels=[0],
    mask=None,
    histSize=[L],
    ranges=[0, L]
)

dst = scaled_img.copy()
multithreshold(dst, x, y)
plt.figure()
plt.bar(range(0, hist.shape[0]), hist.ravel())

show_thresholds(scaled_img, dst)

plt.figure()
ax = plt.subplot(1, 3, 1)
ax.set_title('Original image')
plt.imshow(img2, cmap='gray')
ax = plt.subplot(1, 3, 2)
ax.set_title('Scaled-up image')
plt.imshow(scaled_img, cmap='gray')
np.savetxt('pixel.txt', scaled_img, fmt="%d", delimiter=' ')
ax = plt.subplot(1, 3, 3)
ax.set_title('Threshold image')
plt.imshow(dst, cmap='gray')
cv2.imwrite('threshold_img.jpg', dst)
plt.show()

