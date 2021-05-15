import cv2
import matplotlib.pyplot as plt
import numpy as np

def main(path='1024_noise/2.jpg'):
    L = 256  # number of levels

    img = cv2.imread(path, 0)  # read image in as grayscale

    #Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    plt.figure()
    plt.bar(range(0, hist.shape[0]), hist.ravel())

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
