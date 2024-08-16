# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
plt.style.use('dark_background')
# 이미지 불러오기
image = cv2.imread('papernum1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 이미지 명암 대비를 향상시키는 등의 전처리를 수행
# 예: Adaptive Thresholding을 적용하여 이진화
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# 텍스트가 포함된 영역(ROI)을 선택
# 이 값들은 이미지에 따라 조정해야 합니다. 배번호의 정확한 위치를 찾아 적절히 조정해야 합니다.
# 예시로 x, y, w, h 값은 이미지의 배번호판 위치에 따라 달라집니다.
x, y, w, h = 200, 280, 100, 100  # 실제 윤곽선 감지 기반으로 값을 조정해야 합니다.
roi = thresh[y:y+h, x:x+w]
# OCR 설정
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(roi, config=custom_config)
# 결과 출력
print("Detected text:", text)
# 결과 시각화
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Text')
plt.show()
# %%
# 이미지 불러오기
image = cv2.imread('papernum1.jpg')
# 이미지 크기 조정 (옵션)
image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Otsu의 이진화를 사용하여 이진화
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ROI 설정 (이미지에 따라 조정 필요)
x, y, w, h = 550, 350, 100, 100  # 윤곽선 감지를 통해 얻은 값으로 조정 필요
roi = thresh[y:y+h, x:x+w]
# OCR 설정
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(roi, config=custom_config)
# 결과 출력
print("Detected text:", text.strip())
# 결과 시각화
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Text')
plt.show()
# %%
# 이미지 불러오기
image = cv2.imread('papernum1.png')
# 이미지 크기 조정 (옵션)
image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Otsu의 이진화를 사용하여 이진화
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ROI 설정 (이미지에 따라 조정 필요)
x, y, w, h = 550, 800, 150, 150  # 윤곽선 감지를 통해 얻은 값으로 조정 필요
roi = thresh[y:y+h, x:x+w]
# OCR 설정
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(roi, config=custom_config)
# 결과 출력
print("Detected text:", text.strip())
# 결과 시각화
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Text')
plt.show()
# %%
# 이미지 불러오기
image = cv2.imread('papernum1.png')
# 이미지 크기 조정 (옵션)
# image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
# 이미지의 크기를 확인
height, width = image.shape[:2]
height -= 20
# print(height, width)
# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Otsu의 이진화를 사용하여 이진화
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ROI 설정 (이미지에 따라 조정 필요)
x, y, w, h = 0, 20, width, height-20  # 윤곽선 감지를 통해 얻은 값으로 조정 필요
roi = thresh[y:y+h, x:x+w]
# OCR 설정
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(roi, config=custom_config)
# 결과 출력
print("Detected text:", text.strip())
# 결과 시각화
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Text')
plt.show()
# %%