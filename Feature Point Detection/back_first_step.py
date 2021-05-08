import cv2
from numpy import percentile
import numpy as np

img = cv2.imread('background_2.jpg', cv2.IMREAD_GRAYSCALE)
lower = percentile(img, 0)
upper = percentile(img, 100)
cv2.normalize(img, img, 0, 300+upper, cv2.NORM_MINMAX)
img = cv2.medianBlur(img, 7)
cv2.imwrite('output.jpg', img)
# img = cv2.medianBlur(img,7)
# sub_dst = cv2.subtract(img, 110)
# cv2.imwrite('back2.jpg', sub_dst)