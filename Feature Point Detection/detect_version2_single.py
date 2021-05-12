import cv2
import numpy as np

img = cv2.imread('new_tsmo2.jpg',cv2.IMREAD_COLOR)
# convert to gray scale
img = cv2.medianBlur(img, 5)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30, param1=60, param2=45, minRadius=0, maxRadius=0)
circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 10, param1=60, param2=58, minRadius=0, maxRadius=20)
circles2 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 10, param1=60, param2=51, minRadius=0, maxRadius=20)
#circles2 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 30, param1=120, param2=50, minRadius=0, maxRadius=20)
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 50, param1=240, param2=100, minRadius=0, maxRadius=20)
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 120, param1=120, param2=30, minRadius=0, maxRadius=20)
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 10, param1=60, param2=60, minRadius=0, maxRadius=20)
#circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 50, param1=350, param2=30, minRadius=0, maxRadius=20)
print(circles)
print(circles1)
print(circles2)
if circles is not None :
    circles = np.round(circles[0, :]).astype("int")
if circles1 is not None :
    circles1 = np.round(circles1[0, :]).astype("int")
if circles2 is not None :
    circles2 = np.round(circles2[0, :]).astype("int")
#Draw circles
if (circles is not None) and (len(circles) < 5) :
    for (x,y,r) in circles:
        cv2.circle(img, (x,y), r, (36,255,12), 5)
elif (circles1 is not None) and (len(circles1) < 5):
    for (x,y,r) in circles1:
        cv2.circle(img, (x,y), r, (36,255,12), 5)
elif (circles2 is not None) and (len(circles2) < 5):
    for (x,y,r) in circles2:
        cv2.circle(img, (x,y), r, (36,255,12), 5)
cv2.imwrite('output2.jpg',img)
cv2.imshow('output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()