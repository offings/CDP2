import cv2
import numpy as np
import matplotlib.pyplot as plt

def multithreshold(img):
    thresholds = [24,120]
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

    #print(img.shape) # print pixel size

    min = np.amin(img)
    max = np.amax(img)

    scaled_img = img.copy()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
           scaled_img[i][j] = (img[i][j] - min) / (max - min) * 255

    #Calculate histogram
    hist = cv2.calcHist(
        [scaled_img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    percent_50 = np.percentile(scaled_img, 25)
    print(percent_50)

    dst = scaled_img.copy()
    multithreshold(dst)
    plt.figure()
    plt.bar(range(0, hist.shape[0]), hist.ravel())

    show_thresholds(scaled_img, dst)

    plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.set_title('Original image')
    plt.imshow(img, cmap='gray')
    ax = plt.subplot(1, 3, 2)
    ax.set_title('Scaled-up image')
    plt.imshow(scaled_img, cmap='gray')
    np.savetxt('pixel.txt', scaled_img, fmt = "%d", delimiter=' ')
    ax = plt.subplot(1, 3, 3)
    ax.set_title('Threshold image')
    plt.imshow(dst, cmap='gray')
    cv2.imwrite('threshold_img.jpg',dst)
    plt.show()

if __name__ == '__main__':
    main()