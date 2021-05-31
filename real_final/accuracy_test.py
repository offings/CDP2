import cv2

img=cv2.imread("./real_final/22.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

CANNY_THRESH_1 = 40
CANNY_THRESH_2 = 100

edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

contour_info = []
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

#타원 찾는 부분
contour = contours[1]
contour2 = contours[2]
ellipse = cv2.fitEllipse(contour)
ellipse2 = cv2.fitEllipse(contour2)
cv2.ellipse(img, ellipse, (0, 255, 0), 3)
cv2.ellipse(img, ellipse2, (0, 255, 0), 3)


cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
