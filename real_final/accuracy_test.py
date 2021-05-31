import cv2

def extract_contour(img):
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

    return contours


img=cv2.imread("./real_final/22.jpg")
exp_img=cv2.imread("./expected_result/22.jpg")
contours=extract_contour(img)
exp_contours=extract_contour(exp_img)

#타원 찾는 부분
contour = contours[1]
contour2 = contours[2]
ellipse_1 = cv2.fitEllipse(contour)
ellipse_2 = cv2.fitEllipse(contour2)
cv2.ellipse(img, ellipse_1, (0, 255, 0), 3)
cv2.ellipse(img, ellipse_2, (0, 255, 0), 3)
center_1=ellipse_1[0]
center_2=ellipse_2[0]
print("center_1", center_1, "center_2", center_2)

exp_contour = exp_contours[1]
exp_contour2 = exp_contours[2]
exp_ellipse_1 = cv2.fitEllipse(exp_contour)
exp_ellipse_2 = cv2.fitEllipse(exp_contour2)
cv2.ellipse(exp_img, exp_ellipse_1, (0, 255, 0), 3)
cv2.ellipse(exp_img, exp_ellipse_2, (0, 255, 0), 3)
exp_center_1=exp_ellipse_1[0]
exp_center_2=exp_ellipse_2[0]

final_exp_center_1=(exp_center_1[0]/4, exp_center_1[1]/4)
final_exp_center_2=(exp_center_2[0]/4, exp_center_2[1]/4)

print("exp_center_1", final_exp_center_1, "exp_center_2", final_exp_center_2)


cv2.imshow("result", exp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
