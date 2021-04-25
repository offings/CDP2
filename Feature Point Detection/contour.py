#
# https://076923.github.io/posts/Python-openc
# v-21/
import cv2
import imutils

src = cv2.imread("final.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    cv2.drawContours(src, [contours[i]], 0, (0, 0, 255), 5)

src = imutils.resize(src, width=800)
cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()