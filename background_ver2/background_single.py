import cv2
from numpy import percentile
import numpy as np

# first step : background normalization
img = cv2.imread('image/1024_noise/31.jpg', cv2.IMREAD_GRAYSCALE)
height, width = img.shape
MIN = img.min()
MAX = img.max()
out = img.copy()
low = MIN
high = 60
for j in range(width):
    for k in range(height):
        if img[j][k] < low:
            out[j][k] = 0
        elif img[j][k] > high:
            out[j][k] = high
        # else:
        #     out[j][k] = ((gray[j][k] - MIN) * 255) / (MAX - MIN)
lower = percentile(out, 0)
higher = percentile(out, 100)
cv2.normalize(out, out, lower, higher+50, cv2.NORM_MINMAX)
img = cv2.GaussianBlur(out,(5,5), 0)
cv2.imwrite('first_output_31.jpg', out)


# second step : background remove with contour
#== Parameters
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 100
MASK_DILATE_ITER = 5
MASK_ERODE_ITER = 5
MASK_COLOR = (1.0, 1.0, 1.0) # In BGR format

#-- Read image
img = cv2.imread('first_output_31.jpg')
img2 = cv2.imread('image/1024_noise/31.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#-- Edge detection
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

#-- Find contours in edges, sorqt by area
contour_info = []
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
for c in contours:
    contour_info.append([
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ])
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
#final_contour = [[x for x in contour_info[2][0] if x not in contour_info[0][0]], contour_info[0][1], contour_info[2][2] - contour_info[0][2]]
for lst in contour_info :
    print(lst[2])

final_contour = contour_info[2]
# final_cntour = contour_info[2]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, final_contour[0], (255, 255, 255))

#-- Smooth mask, then blur it
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#-- Blend masked img into MASK_COLOR background
mask_stack = mask_stack.astype('float32') / 255.0
img2 = img2.astype('float32') / 255.0
masked = (mask_stack * img2) + ((1-mask_stack) * MASK_COLOR)
masked = (masked * 255).astype('uint8')

cv2.waitKey()
cv2.imwrite('second_output_31.jpg',masked)