import cv2
from numpy import percentile
img = cv2.imread('background.jpg', cv2.IMREAD_GRAYSCALE)
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
lower=percentile(img, 0)
upper=percentile(img,100)
cv2.normalize(img, img, 0, 700-upper, cv2.NORM_MINMAX) # tune parameters

cv2.imwrite('back3.jpg', img)