import cv2
import tsmo
import matplotlib.pyplot as plt


def show_thresholds(src_img, dst_img, thresholds):
    # large_colors = [0, 51, 80, 102, 130,
    #           153, 170, 204,210,220,230,235,240,245,250,
    #           255]

    colors=[0]

#    small_colors=[0,60,120,190,255]
    masks = tsmo.multithreshold(src_img, thresholds)
    print(len(masks))
    mask_len=len(masks)




    for i in range(1, mask_len+1) :

        c_len=int(255/mask_len*i)
        colors.append(c_len)


    # if len(masks)>10:
    #     for i, mask in enumerate(masks):
    #         dst_img[mask] = large_colors[i]
    #
    # elif len(masks)>5:
    #     for i, mask in enumerate(masks):
    #         dst_img[mask]=colors[i]
    # else:
    for i, mask in enumerate(masks):
        dst_img[mask]=colors[i]

    return dst_img

def main(path='../Feature Point Detection/second_step/7.jpg'):
    L = 256  # number of levels
    M = 128  # number of bins for bin-grouping normalisation

    img = cv2.imread(path, 0)  # read image in as grayscale

    # Blur image to denoise
 #   img = cv2.GaussianBlur(img, (5, 5), 0)

    # Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    thresholds = tsmo.modified_TSMO(hist, M=M, L=L)

   # img=img*1.8
    avg = img.sum() / img.size / 2.0
    img = img * 2.5 - avg

    #    img_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_1=img.copy()
    img_auto = img_1.copy()

    show_thresholds(img, img_1, thresholds[0:3])
    show_thresholds(img, img_auto, thresholds[0:7])

    plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.set_title('Original image')
    plt.imshow(img, cmap='gray')
    ax = plt.subplot(1, 3, 2)
    ax.set_title('{} levels (Automatic)'.format(len(thresholds)))
    cv2.imwrite('third_result/7.jpg', img_auto)
    plt.imshow(img_auto, cmap='gray')
    ax = plt.subplot(1, 3, 3)
    ax.set_title('3 levels')
    plt.imshow(img_1, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
