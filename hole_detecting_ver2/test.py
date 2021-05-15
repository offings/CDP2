import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def multithreshold(img):
    thresholds = [115]
    masks = np.zeros((len(thresholds) + 1, img.shape[0], img.shape[1]), bool)
    for i, t in enumerate(sorted(thresholds)):
        masks[i+1] = (img > t)
    masks[0] = ~masks[1]
    for i in range(1, len(masks) - 1):
        masks[i] = masks[i] ^ masks[i+1]
    return masks

def show_thresholds(src_img, dst_img):
    colors=[0]
    masks = multithreshold(src_img)
    mask_len=len(masks)
    mask_len=mask_len-1

    for i in range(1, mask_len+1) :
        c_len=int(255/mask_len*(i))
        colors.append(c_len)

    for i, mask in enumerate(masks):
        dst_img[mask]=colors[i]

    return dst_img

def main(path='cropped.jpg'):
    L = 256  # number of levels

    img = cv2.imread(path, 0)  # read image in as grayscale

    #Calculate histogram



    res=cv2.equalizeHist(img)
    np.savetxt('pixel.txt', res, fmt="%d", delimiter=' ')

    show_thresholds(res, res)

    hist = cv2.calcHist(
        [res],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )
    plt.figure()
    plt.bar(range(0, hist.shape[0]), hist.ravel())


    plt.figure()
    plt.imshow(res, cmap='gray')
    plt.show()







if __name__ == '__main__':
    main()
