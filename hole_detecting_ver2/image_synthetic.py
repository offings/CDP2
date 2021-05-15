import cv2
daum_logo = cv2.imread('daum.png', 1)
img1 = cv2.imread('b.jpg', 1)
h, w, c = daum_logo.shape
roi = img1[150:150+h, 150:150+w]#배경이미지의 변경할(다음 로고 넣을) 영역
mask = cv2.cvtColor(daum_logo, cv2.COLOR_BGR2GRAY)#로고를 흑백처리
#이미지 이진화 => 배경은 검정. 글자는 흰색
mask[mask[:]==255]=0
mask[mask[:]>0]=255
mask_inv = cv2.bitwise_not(mask) #mask반전.  => 배경은 흰색. 글자는 검정
daum = cv2.bitwise_and(daum_logo, daum_logo, mask=mask)#마스크와 로고 칼라이미지 and하면 글자만 추출됨
back = cv2.bitwise_and(roi, roi, mask=mask_inv)#roi와 mask_inv와 and하면 roi에 글자모양만 검정색으로 됨
dst = cv2.add(daum, back)#로고 글자와 글자모양이 뚤린 배경을 합침
img1[150:150+h, 150:150+w] = dst  #roi를 제자리에 넣음
# cv2.imshow('mask', dst)
cv2.imshow('img1', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()