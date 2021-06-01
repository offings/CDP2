import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def accuracy_test_ver2() :
    index_list = []
    # first_x = []
    # first_y = []
    # second_x = []
    # second_y = []
    for index in range(1, 34):
        print('index', index)
        index_list.append(index)
        img = cv2.imread('expected_result/{}.jpg'.format(index))
        img1 = img.copy()
        img2 = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res, thr = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(2):
            if(index == 28): cnt = contours[i+1]
            else: cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            box = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img1, [box], 0, (0, 255, 0), 3)

            ellipse = cv2.fitEllipse(cnt)
            # if i == 0 :
            #     first_x.append(ellipse[0][0] / 4)
            #     first_y.append(ellipse[0][1] / 4)
            # else :
            #     second_x.append(ellipse[0][0] / 4)
            #     second_y.append(ellipse[0][1] / 4)

            print('center x : {}, center y : {},'.format(ellipse[0][0] / 4, ellipse[0][1] / 4))
            cv2.ellipse(img2, ellipse, (0, 255, 0), 2)

        images = [img1, img2]
        titles = ['Rectangle', 'Circle and Ellipse']
        plt.Figure(figsize=(8, 8))
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(titles[i])
            plt.imshow(images[i])
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    # data = {'index' : index_list, 'expected x(1)' : first_x, 'expected y(1)' : first_y, 'expected x(2)' : second_x, 'expected y(2)' : second_y}
    # df = pd.DataFrame(data)
    # df.to_csv('accuracy_test.csv', index=False, mode = 'w')

accuracy_test_ver2()