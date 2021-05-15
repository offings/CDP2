import cv2

crop = cv2.imread('result.jpg', 0)
img = cv2.imread('contour.jpg', 0)
img = cv2.resize(img, dsize = (800, 800), interpolation=cv2.INTER_AREA)

# shape[0] = height, shape[1] = width
height, width = crop.shape
print(height, width)
roi = img[396 : 396 + height, 411 : 411 + width]
img[396 : 396 + height, 411 : 411 + width] = crop
cv2.imwrite('contour.jpg', img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
