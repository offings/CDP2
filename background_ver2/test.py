import cv2
import matplotlib.pyplot as plt
import numpy as np

def main(path='../Feature Point Detection/1024_noise/3.jpg'):

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read image in as grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    MIN = gray.min()
    MAX = gray.max()
    out = gray.copy()
    low = MIN
    high = int(input("higher value : "))
    for i in range(width) :
        for j in range(height) :
            if gray[i][j] < low :
                out[i][j] = 0
            elif gray[i][j] > high :
                out[i][j] = 255
            else :
                out[i][j] = ((gray[i][j] - MIN) * 255) / (MAX - MIN)
    cv2.imshow('orignal', gray)
    cv2.imshow('end_in', out)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.subplot(1, 2, 2)
    plt.hist(out.ravel(), 256, [0, 256])
    plt.show()

    cv2.waitKey()

if __name__ == '__main__':
    main()
