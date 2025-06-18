import cv2

img = cv2.imread('E:\LF\LF_GrabImage\GrabImage_00_R.bmp', cv2.IMREAD_GRAYSCALE)

height, width = img.shape[:2]

print(f"Original size: {width} x {height}")

resized = cv2.resize(img, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

cv2.imwrite('E:\LF\LF_GrabImage\esized_image.bmp', resized)
print("Resized image saved.")