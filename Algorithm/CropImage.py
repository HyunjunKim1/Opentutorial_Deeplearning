import cv2

img = cv2.imread('E:\LF\LF_GrabImage\GrabImage_00.bmp', cv2.IMREAD_GRAYSCALE)

height, width = img.shape[:2]

tile_height = 6347

for i in range(7):
    y_start = i * tile_height
    y_end = y_start + tile_height
    tile = img[y_start:y_end, :]
    cv2.imwrite(f'E:\LF\LF_GrabImage\Tile_{i+1}.bmp', tile)
