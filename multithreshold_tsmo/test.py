import cv2
import tsmo
import matplotlib.pyplot as plt


def show_thresholds(src_img, dst_img, thresholds):
    large_colors = [(0, 0, 0), (51, 51, 51), (80, 80, 80), (102, 102, 102), (130, 130, 130),
              (153, 153, 153), (170, 170, 170), (204, 204, 204),(210,210,210),(220,220,220),
              (230, 230, 230),(235,235,235),(240,240,240),(245,245,245),(250,250,250),
              (255, 255, 255)]

    colors=[(0, 0, 0), (51, 51, 51), (80, 80, 80), (102, 102, 102), (130, 130, 130),
              (153, 153, 153), (170, 170, 170), (204, 204, 204), (230, 230, 230),
              (255, 255, 255)]

    small_colors=[(0,0,0),(60,60,60),(120,120,120),(190,190,190),(255,255,255)]
    masks = tsmo.multithreshold(src_img, thresholds)
    print(len(masks))

    if len(masks)>10:
        for i, mask in enumerate(masks):
            dst_img[mask] = large_colors[i]

    elif len(masks)>5:
        for i, mask in enumerate(masks):
            dst_img[mask]=colors[i]
    else:
        for i, mask in enumerate(masks):
            dst_img[mask]=small_colors[i]

    return dst_img

def main(path='../Feature Point Detection/second_step/33.jpg'):
    L = 256  # number of levels
    M = 64  # number of bins for bin-grouping normalisation

    img = cv2.imread(path, 0)  # read image in as grayscale

    # Blur image to denoise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    thresholds = tsmo.modified_TSMO(hist, M=M, L=L)



    img_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_auto = img_1.copy()

    show_thresholds(img, img_1, thresholds[0:3])
    show_thresholds(img, img_auto, thresholds)

    plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.set_title('Original image')
    plt.imshow(img, cmap='gray')
    ax = plt.subplot(1, 3, 2)
    ax.set_title('{} levels (Automatic)'.format(len(thresholds)))
    cv2.imwrite('first_result/output_33.jpg', img_auto)
    plt.imshow(img_auto)
    ax = plt.subplot(1, 3, 3)
    ax.set_title('3 levels')
    plt.imshow(img_1)
    plt.show()

if __name__ == '__main__':
    main()
