import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def accuracy_test_ver2() :
    for index in range(1, 34):
        print('index', index)
        img = cv2.imread('real_final/{}.jpg'.format(index))
        img1 = img.copy()
        img2 = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res, thr = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(2):
            if index == 23 and i == 0: cnt = contours[13]
            elif index == 23 and i == 1: cnt = contours[15]
            if index == 24 and i == 0: cnt = contours[9]
            elif index == 24 and i == 1: cnt = contours[10]
            if index == 31 and i == 0: cnt = contours[66]
            elif index == 31 and i == 1: cnt = contours[67]

            x, y, w, h = cv2.boundingRect(cnt)
            box = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img1, [box], 0, (0, 255, 0), 3)

            ellipse = cv2.fitEllipse(cnt)
            print(ellipse[0][0], ellipse[0][1], ellipse[0][0], ellipse[0][1])
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

accuracy_test_ver2()