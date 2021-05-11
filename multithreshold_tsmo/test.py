import cv2
import tsmo
import matplotlib.pyplot as plt
import numpy as np


def show_thresholds(src_img, dst_img, thresholds):

    colors=[0]

    masks = tsmo.multithreshold(src_img, thresholds)
    mask_len=len(masks)
    mask_len=mask_len-1

    for i in range(1, mask_len+1) :
        c_len=int(255/mask_len*(i))
        colors.append(c_len)

    for i, mask in enumerate(masks):
        dst_img[mask]=colors[i]

    return dst_img

def main(path='../Feature Point Detection/second_step/24.jpg'):
    L = 256  # number of levels
    M = 64  # number of bins for bin-grouping normalisation

    img = cv2.imread(path, 0)  # read image in as grayscale

    # Blur image to denoise
 #

    #
    # def CLAHE(img, limit=2.0, grid=8):
    #     clahe=cv2.createCLAHE(clipLimit=limit, tileGridSize=(grid, grid))
    #     return clahe.apply(img)
    #
    # img=CLAHE(img)


    #Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    thresholds = tsmo.modified_TSMO(hist, M=M, L=L)


   # img = img * 1.5
 #   img = cv2.GaussianBlur(img, (5, 5), 0)

    avg = img.sum() / img.size / 2.0
    img = img * 2.5 - avg
    #
    # alpha=-0.5
    # img=np.clip(img+(img-128.)*alpha, 0, 255).astype(np.uint8)
    #
    # kernel=np.ones((6,6), np.uint8)
    # img=cv2.morphologyEx(img, cv2.MORPH_OEPN, kernel)
    #
    #    img_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_1=img.copy()
    img_auto = img_1.copy()

    show_thresholds(img, img_1, thresholds[0:3])
    show_thresholds(img, img_auto, thresholds[0:8])



    plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.set_title('Original image')
    plt.imshow(img, cmap='gray')
    ax = plt.subplot(1, 3, 2)
    ax.set_title('{} levels (Automatic)'.format(len(thresholds)))
    cv2.imwrite('result_4/24.jpg', img_auto)
    plt.imshow(img_auto, cmap='gray')
    ax = plt.subplot(1, 3, 3)
    ax.set_title('3 levels')
    plt.imshow(img_1, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
