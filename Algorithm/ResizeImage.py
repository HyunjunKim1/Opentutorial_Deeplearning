import cv2

# 이미지 로드 (Grayscale)
img = cv2.imread(r'E:\LF\LF_GrabImage\GrabImage_00.bmp', cv2.IMREAD_GRAYSCALE)

# 원본 크기
height, width = img.shape[:2]
print(f"Original size: {width} x {height}")

# 목표 너비
target_width = 9344

# 비율 계산
scale = target_width / width
target_height = int(height * scale)

# 리사이즈
resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

# 저장
cv2.imwrite(r'E:\LF\LF_GrabImage\resized_image.bmp', resized)
print(f"Resized image saved to {target_width} x {target_height}")
