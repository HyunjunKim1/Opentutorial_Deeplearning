import cv2
import numpy as np

img = cv2.imread(r'd:\\TEST\\Test2.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

h, w = binary.shape

# 좌상단: 처음 흰색
def find_first_white_left_to_right_top_down():
    for x in range(w):
        for y in range(h):
            if binary[y, x] == 255:
                return (x, y)
    return None

# 우상단: 마지막 흰색
def find_last_white_right_to_left_top_down():
    for x in reversed(range(w)):
        for y in range(h):
            if binary[y, x] == 255:
                return (x, y)
    return None

# 좌하단: 마지막 흰색 (← 아래에서 위, 좌에서 우)
def find_last_white_left_to_right_bottom_up():
    for x in range(w):
        for y in reversed(range(h)):
            if binary[y, x] == 255:
                return (x, y)
    return None

# 우하단: 마지막 흰색
def find_last_white_right_to_left_bottom_up():
    for x in reversed(range(w)):
        for y in reversed(range(h)):
            if binary[y, x] == 255:
                return (x, y)
    return None

# 찾기
tl = find_first_white_left_to_right_top_down()
tr = find_last_white_right_to_left_top_down()
bl = find_last_white_left_to_right_bottom_up()
br = find_last_white_right_to_left_bottom_up()

# ROI 계산
if None not in [tl, tr, bl, br]:
    x_min = min(tl[0], bl[0])
    x_max = max(tr[0], br[0])
    y_min = min(tl[1], tr[1])
    y_max = max(bl[1], br[1])

    roi = img[y_min:y_max, x_min:x_max]
    zoomed = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Zoomed", zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ dot의 4 모서리를 찾지 못했습니다.")