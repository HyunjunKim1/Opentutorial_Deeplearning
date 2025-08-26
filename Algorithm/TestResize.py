import cv2
import numpy as np

# 1. 이미지 불러오기
img = cv2.imread(r'E:\LF\LF_GrabImage\GrabImage_00.bmp', cv2.IMREAD_GRAYSCALE)

# 2. 원하는 세로 크기
target_height = 21426
original_height, original_width = img.shape

# 3. 세로 크기에 맞춰 비율 유지한 가로 크기 계산
aspect_ratio = original_width / original_height
new_width = int(target_height * aspect_ratio)

resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)

# 4. 좌우 여백 계산
target_width = 9344
padding = target_width - new_width

if padding < 0:
    raise ValueError("패딩을 넣을 수 없습니다. new_width가 target_width보다 큽니다.")

left_pad = padding // 2
right_pad = padding - left_pad

# 5. 좌우 여백을 넣어 전체 이미지 크기를 9344로 맞춤
padded_img = cv2.copyMakeBorder(
    resized_img,
    top=0,
    bottom=0,
    left=left_pad,
    right=right_pad,
    borderType=cv2.BORDER_CONSTANT,
    value=0  # 검정색
)

# 6. 저장
cv2.imwrite(r'E:\LF\LF_GrabImage\resized_image.bmp', padded_img)
print("완료: 최종 해상도 =", padded_img.shape[::-1])  # (width, height)