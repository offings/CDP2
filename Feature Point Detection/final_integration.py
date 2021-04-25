import cv2
import numpy as np
import math

# 이미지에서 선을 찾는 함수 정의
def contour() :
    src = cv2.imread("test.jpg", cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # cv2.threshold(src, thresh, maxval, type)
    # src : input image로 single-channel 이미지(grayscale 이미지)
    # thresh : 임계값
    # maxval : 입계값을 넘었을 때 적용할 value
    # type : thresholding type
    # cv2.THRESH_BINARY: threshold보다 크면 value이고 아니면 0으로 바꾸어 줍니다.
    # cv2.THRESH_BINARY_INV: threshold보다 크면 0이고 아니면 value로 바꾸어 줍니다.
    # cv2.THRESH_TRUNC: threshold보다 크면 value로 지정하고 작으면 기존의 값 그대로 사용한다.
    # cv2.THRESH_TOZERO: treshold_value보다 크면 픽셀 값 그대로 작으면 0으로 할당한다.
    # cv2.THRESH_TOZERO_INV: threshold_value보다 크면 0으로 작으면 그대로 할당해준다.
    # return값 : 1) ret - thresholding에 사용한 threshold, 2) binary - thresholding이 적용된 binary image
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    # 이미지의 윤곽선을 검출하기 위해 cv2.findContours 이용
    # cv2.findContours()를 이용하여 binary 이미지에서 윤곽선(contour)검색
    # cv2.findContours(binary, 검색 방법, 근사화 방법)
    # 검색 방법
    # cv2.RETR_EXTERNAL : 외곽 윤곽선만 검출하며, 계층 구조를 구성하지 않습니다.
    # cv2.RETR_LIST : 모든 윤곽선을 검출하며, 계층 구조를 구성하지 않습니다.
    # cv2.RETR_CCOMP : 모든 윤곽선을 검출하며, 계층 구조는 2단계로 구성합니다.
    # cv2.RETR_TREE : 모든 윤곽선을 검출하며, 계층 구조를 모두 형성합니다. (Tree 구조)
    # 근사화 방법
    # cv2.CHAIN_APPROX_NONE : 윤곽점들의 모든 점을 반환합니다.
    # cv2.CHAIN_APPROX_SIMPLE : 윤곽점들 단순화 수평, 수직 및 대각선 요소를 압축하고 끝점만 남겨 둡니다.
    # cv2.CHAIN_APPROX_TC89_L1 : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.
    # cv2.CHAIN_APPROX_TC89_KCOS : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.
    # return값 : 1) contours - 윤곽선, 2) hierarchy - 계층 구조
    # contoures : 윤곽선의 지점
    # hierarchy : 윤곽선의 계층 구조
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        # 각 윤곽선을 그림.
        cv2.drawContours(src, [contours[i]], 0, (0, 0, 255), 5)

    cv2.imwrite("output.jpg", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지에서 원/점을 찾는 함수 정의
def detect_circle() :
    img = cv2.imread('test.jpg', 0)
    # noise를 제거하기 위해 blur 진행
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 허프 변환을 이용하여 원을 찾기 위해 cv2.HoughCircles 함수 사용
    # cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, ratio, minval, param1, param2, minRadius, maxRadius)
    # cv2.HOUGH_GRADIENT : 허프 변환을 이용하여 원을 찾는 방법, 현재 하나 밖에 없음
    # ratio : 원본 이미지와 허프변환 카운팅 결과 이미지의 ratio, 대부분 1로 설정
    # minval : 찾은 원들의 중심간 최소 거리. 중심간의 거리가 이 값보다 작으면 나중에 찾은 원은 무시합니다.
    # param1 : cv2.HoughCircles() 함수는 내부적으로 Canny Edge Detection을 사용하는데 Canny() 함수의 인자로 들어가는 maxVal 값
    # param2 : 원으로 판단하는 허프 변환 카운팅 값. 값이 너무 작으면 워하지 않는 많은 원들이, 값이 너무 크면 원을 못 찾을 수 있으니 적절한 값을 주어야 함.
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=60, param2=60, minRadius=0, maxRadius=0)
    # circles의 값들을 반올림/반내림하고 UINT16으로 변환
    circles = np.uint16(np.around(circles))
    for i in circles[0, :] :
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.imwrite('output2.jpg', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지에서 각도를 찾는 함수 정의
def line_angle() :
    img_before = cv2.imread('test.jpg')

    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    # Canny Edge Detection 알고리즘 구현 함수
    # cv2.Canny(img, min, max)
    # img : Canny Edge Detection을 수행할 원본 Grayscale 이미지
    # min : minimum thresholding value
    # max : maximum thresholding value
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)

    # HoughLines를 통해 이미지의 선 감지
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)

    print("Angle is {}".format(median_angle))

if __name__ == '__main__' :
    number = int(input("번호를 선택하세요!\n 1. 선 / 2. 점 / 3. 각도 : "))
    while(number != 4) :
        if number == 1 :
            contour()
        elif number == 2 :
            detect_circle()
        else :
            line_angle()
        number = int(input("번호를 선택하세요!\n 1. 선 / 2. 점 / 3. 각도 : "))