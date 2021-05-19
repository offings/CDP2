import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import percentile
index = 0
#define path
def path(num):
    if num == 1 :
        return '../background_ver2/image/first_output/{}.jpg'.format(index)
    elif num == 2 :
        return '../background_ver2/image/second_output/{}.jpg'.format(index)
    elif num == 3 :
        return 'final_image_mid/{}.jpg'.format(index)

def detect_screw():
    img = cv2.imread(path(2))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lower = percentile(gray, 0)
    higher = percentile(gray, 100)

    cv2.normalize(gray, gray, lower - 100, higher + 100, cv2.NORM_MINMAX)

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

    x_screw = []
    y_screw = []

    for i in range(len(contour_info[0][0])):
        x, y = contour_info[0][0][i][0]
        x_screw.append(x)
        y_screw.append(y)
        zip_list = zip(x_screw, y_screw)

    screw = list(zip_list).copy()
    print(screw)

    cv2.imwrite("contour.jpg", img)
    return screw

def multithreshold(img, x, x_v2):
    thresholds = []
    thresholds.append(x)
    thresholds.append(x_v2)

    masks = np.zeros((len(thresholds) + 1, img.shape[0], img.shape[1]), bool)
    for i, t in enumerate(sorted(thresholds)):
        masks[i+1] = (img > t)
    masks[0] = ~masks[1]
    for i in range(1, len(masks) - 1):
        masks[i] = masks[i] ^ masks[i+1]
    return masks

def show_thresholds(src_img, dst_img):
    colors=[0]
    masks = multithreshold(src_img, x, x_v2)
    mask_len=len(masks)
    mask_len=mask_len-1

    for i in range(1, mask_len+1) :
        c_len=int(255/mask_len*(i))
        colors.append(c_len)

    for i, mask in enumerate(masks):
        dst_img[mask]=colors[i]

    return dst_img

# main start
for index in range(1, 34):
    print('index', index)
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
    x_list_v2 = []
    y_list_v2 = []

    # 원하는 area size만큼의 contour들을 draw하고 해당 index와 area size를 출력하는 부분
    for area in range(len(contour_info)) :
        if(contour_info[area][2] <= 1100.0 and contour_info[area][2] >= 200.0) :
            cv2.drawContours(img, contour_info[area], 0, (0, 0, 255), 5)
            # print(area, contour_info[area][0])
            re_area += 1
            if (re_area == 1) :
                for i in range(len(contour_info[area][0])):
                    x,y=contour_info[area][0][i][0]
                    #print(x, y)
                    x_list.append(x)
                    y_list.append(y)
                half_index = int(0.5 * (len(contour_info[area][0])))
                x_origin, y_origin = contour_info[area][0][0][0]
                x_end = x_list[half_index]
                y_end = y_list[half_index]
                center_x = int((x_origin + x_end)/2)
                center_y = int((y_origin + y_end)/2)
                cir_x = center_x - 17
                print('x_origin : {}, y_origin : {}, x_end : {}, y_end : {}' .format(x_origin, y_origin, x_end, y_end))
                print('center_x : {}, center_y : {}'.format(center_x, center_y))

            elif(re_area == 3) :
                for i in range(len(contour_info[area][0])):
                    x,y=contour_info[area][0][i][0]
                    #print(x, y)
                    x_list_v2.append(x)
                    y_list_v2.append(y)

                half_index_v2 = int(0.5 * (len(contour_info[area][0])))
                x_origin_v2, y_origin_v2 = contour_info[area][0][0][0]
                x_end_v2 = x_list_v2[half_index_v2]
                y_end_v2 = y_list_v2[half_index_v2]
                center_x_v2 = int((x_origin_v2 + x_end_v2) / 2)
                center_y_v2 = int((y_origin_v2 + y_end_v2) / 2)
                cir_x_v2 = center_x_v2 - 17
                print('x_origin_v2 : {}, y_origin_v2 : {}, x_end_v2 : {}, y_end_v2 : {}'.format(x_origin_v2, y_origin_v2, x_end_v2, y_end_v2))
                print('center_x_v2 : {}, center_y_v2 : {}'.format(center_x_v2, center_y_v2))


    screw_result = []
    screw_result = detect_screw()

    y_mid = 0
    for i in range(len(screw_result)):
        if(screw_result[i][0] == x_end and screw_result[i][1] > y_end):
            x_line = x_end
            y_line = screw_result[i][1]
            y_mid = int((y_line + y_end) / 2)

        elif(screw_result[i][0] == x_end_v2 and screw_result[i][1] > y_end_v2):
            x_line_v2 = x_end_v2
            y_line_v2 = screw_result[i][1]
            y_mid_v2 = int((y_line_v2 + y_end_v2) / 2)

    cv2.imwrite("contour.jpg", img)

    L = 256  # number of levels
    img2 = cv2.imread(path(2), 0)  # read image in as grayscale

    min = np.amin(img2)
    max = np.amax(img2)
    scaled_img = img2.copy()
    for i in range(0, img2.shape[0]):
        for j in range(0, img2.shape[1]):
            scaled_img[i][j] = (img2[i][j] - min) / (max - min) * 255

    temp1 = scaled_img[y_origin][x_origin]
    temp2 = scaled_img[y_end][x_end]
    temp1_v2 = scaled_img[y_origin_v2][x_origin_v2]
    temp2_v2 = scaled_img[y_end_v2][x_end_v2]

    # x = int((temp1 + temp3)/2)
    # y = int((temp2 + temp3)/2)
    # x_v2 = int((temp1_v2 + temp3_v2)/2)
    # y_v2 = int((temp2_v2 + temp3_v2)/2)
    x = int((temp1 + temp2)/2)
    x_v2 = int((temp1_v2 + temp2_v2)/2)
    # Calculate histogram
    hist = cv2.calcHist(
        [scaled_img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    dst = scaled_img.copy()
    multithreshold(dst, x, x_v2)
    print('thresholds : ', x, x_v2)
    #plt.figure()
    #plt.bar(range(0, hist.shape[0]), hist.ravel())

    show_thresholds(scaled_img, dst)

    # plt.figure()
    # ax = plt.subplot(1, 3, 1)
    # ax.set_title('Original image')
    # plt.imshow(img2, cmap='gray')
    # ax = plt.subplot(1, 3, 2)
    # ax.set_title('Scaled-up image')
    # plt.imshow(scaled_img, cmap='gray')
    # #np.savetxt('pixel.txt', scaled_img, fmt="%d", delimiter=' ')
    # ax = plt.subplot(1, 3, 3)
    # ax.set_title('Threshold image')
    # plt.imshow(dst, cmap='gray')
    cv2.imwrite(path(3), dst)
    # plt.show()