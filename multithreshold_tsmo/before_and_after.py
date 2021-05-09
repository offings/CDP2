import cv2, numpy as np
import matplotlib.pylab as plt

img1 = cv2.imread('4096/2_HR_X = 351.2425_Y =353.2472_Z =91.99998.jpg')
img2 = cv2.imread('first_result/output_2.jpg')
img3 = cv2.imread('../Feature Point Detection/second_step/2.jpg')

imgs = [img1, img2, img3]
hists = []
for i, img in enumerate(imgs):
    plt.subplot(1, len(imgs), i + 1)
    plt.title('img%d' % (i + 1))
    plt.axis('off')
    plt.imshow(img[:, :, ::-1])
    # 각 이미지를 HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # H,S 채널에 대한 히스토그램 계산
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 0~1로 정규화
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    hists.append(hist)

# cv2.HISTCMP_CORREL: 상관관계 (1: 완전 일치, -1: 완전 불일치, 0: 무관계)
# cv2.HISTCMP_CHISQR: 카이제곱 (0: 완전 일치, 무한대: 완전 불일치)
# cv2.HISTCMP_INTERSECT: 교차 (1: 완전 일치, 0: 완전 불일치 - 1로 정규화한 경우)
query = hists[0]
methods = {'CORREL': cv2.HISTCMP_CORREL, 'CHISQR': cv2.HISTCMP_CHISQR,
           'INTERSECT': cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA': cv2.HISTCMP_BHATTACHARYYA}

for j, (name, flag) in enumerate(methods.items()):
    print('%-10s' % name, end='\t')
    for i, (hist, img) in enumerate(zip(hists, imgs)):
        # 각 메서드에 따라 img1과 각 이미지의 히스토그램 비교
        ret = cv2.compareHist(query, hist, flag)
        if flag == cv2.HISTCMP_INTERSECT:  # 교차 분석인 경우
            ret = ret / np.sum(query)  # 비교대상으로 나누어 1로 정규화
        print("img%d:%7.2f" % (i + 1, ret), end='\t')
    print()
plt.show()
