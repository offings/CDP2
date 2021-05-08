import cv2
from numpy import percentile
import numpy as np

for i in range(33) :
    # first step : background normalization
    img = cv2.imread('1024_noise/'+str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE)
    lower = percentile(img, 0)
    upper = percentile(img, 100)
    if (i==1) or (i==5) or (i==23) :
        cv2.normalize(img, img, 0, 250+upper, cv2.NORM_MINMAX)
    elif (i==20) or (i==30) :
        img = cv2.add(img, 70)
        cv2.normalize(img, img, 0, 300+upper, cv2.NORM_MINMAX)
    else :
        cv2.normalize(img, img, 0, 300+upper, cv2.NORM_MINMAX)
    img = cv2.medianBlur(img, 7)
    cv2.imwrite('first_step/'+str(i+1)+'.jpg', img)

    # second step : background remove with contour
    #== Parameters
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 100
    MASK_DILATE_ITER = 5
    MASK_ERODE_ITER = 5
    MASK_COLOR = (1.0,1.0,1.0) # In BGR format

    #-- Read image
    img = cv2.imread('first_step/'+str(i+1)+'.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #-- Edge detection
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #-- Find contours in edges, sort by area
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
    if (i==30) :
        final_contour = contour_info[1]
    else :
        final_contour = contour_info[2]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, final_contour[0], (255))

    #-- Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    #-- Blend masked img into MASK_COLOR background
    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0
    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')

    cv2.waitKey()
    cv2.imwrite("second_step/"+str(i+1)+'.jpg',masked)