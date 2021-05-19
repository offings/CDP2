import cv2
import matplotlib.pyplot as plt
import numpy as np

#define path
def path(num):
    if num == 1 :
        return '../background_ver2/image/first_output/22.jpg'
    if num == 2 :
        return '../background_ver2/image/second_output/22.jpg'

def detect_screw():
    img = cv2.imread(path(2))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

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
    cv2.drawContours(img, contour_info[0], 0, (0, 0, 255), 5)

    x_screw = []
    y_screw = []

    for i in range(len(contour_info[0][0])):
        x, y = contour_info[0][0][i][0]
        x_screw.append(x)
        y_screw.append(y)
        zip_list = zip(x_screw, y_screw)
    # print(list(zip_list))

    screw = list(zip_list).copy()
    # print(screw)

    cv2.imwrite("contour.jpg", img)
    return screw

def multithreshold(img, x, y):
    thresholds = []
    thresholds.append(x)
    thresholds.append(y)

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

# main start
img = cv2.imread(path(1))
out = cv2.imread(path(2))
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
        cv2.drawContours(img, contour_info[area], 0, (0, 0, 255), 5)
        re_area += 1
        if (re_area == 2):
            for i in range(len(contour_info[area][0])):
                x,y=contour_info[area][0][i][0]
                #print(x, y)
                x_list.append(x)
                y_list.append(y)

            x_min, y_min = x_list[np.argmin(y_list)], min(y_list)
            x_max, y_max = x_list[np.argmax(y_list)], max(y_list)
            center_x = int((x_min + x_max)/2)
            center_y = int((y_min + y_max)/2)
            cir_x = center_x - 18
            print('x_max : {}, y_max : {}, x_min : {}, y_min : {}' .format(x_max, y_max, x_min, y_min))
            print('center_x : {}, center_y : {}'.format(center_x, center_y))
            break

screw_result = []
screw_result = detect_screw()

y_mid = 0
for i in range(len(screw_result)):
    if(screw_result[i][0] == x_max and screw_result[i][1] > y_max):
        x_line = x_max
        y_line = screw_result[i][1]
        y_mid = int((y_line + y_max) / 2)

cv2.imwrite("contour.jpg", img)

L = 256  # number of levels
img2 = cv2.imread(path(2), 0)  # read image in as grayscale

min = np.amin(img2)
max = np.amax(img2)
scaled_img = img2.copy()
for i in range(0, img2.shape[0]):
    for j in range(0, img2.shape[1]):
        scaled_img[i][j] = (img2[i][j] - min) / (max - min) * 255

temp1 = scaled_img[center_y][center_x]
temp2 = scaled_img[center_y][cir_x]
temp3 = scaled_img[y_mid][x_max]
print('temp1 : {}, temp2 : {}, temp3 : {}'.format(temp1, temp2, temp3))
#print(center_y, center_x, cir_x, y_mid, x_origin)

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
print('thresholds : ', x, y)
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
#np.savetxt('pixel.txt', scaled_img, fmt="%d", delimiter=' ')
ax = plt.subplot(1, 3, 3)
ax.set_title('Threshold image')
plt.imshow(dst, cmap='gray')
cv2.imwrite('threshold_img.jpg', dst)
plt.show()

