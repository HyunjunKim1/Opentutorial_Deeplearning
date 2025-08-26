import cv2
import numpy as np

# 이미지 로딩
img = cv2.imread(r'D:\\ImageMatrix\\roi.png', cv2.IMREAD_GRAYSCALE)

# 블러로 노이즈 제거
blur = cv2.medianBlur(img, 5)

# 원 검출
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=5,            # dot 간 최소 거리 (조정 필요)
    param1=100,
    param2=10,            # 작을수록 더 민감하게 감지
    minRadius=1,
    maxRadius=6
)

# 중심 좌표만 추출
circles = np.uint16(np.around(circles))
pts = circles[0][:, :2].astype(np.float32)

print(f"[INFO] Detected {len(pts)} dots")

# 축별 클러스터링 함수
def cluster_axis(pts_1d, tolerance=3):
    clusters = []
    for p in sorted(pts_1d):
        for cluster in clusters:
            if abs(cluster[-1] - p) < tolerance:
                cluster.append(p)
                break
        else:
            clusters.append([p])
    return [np.mean(c) for c in clusters]

x_clusters = cluster_axis(pts[:, 0])
y_clusters = cluster_axis(pts[:, 1])

# 빈 그리드 생성
grid = np.ones((len(y_clusters), len(x_clusters)), dtype=np.uint8) * 255

# 점을 그리드에 매핑
for pt in pts:
    x_idx = np.argmin([abs(pt[0] - xc) for xc in x_clusters])
    y_idx = np.argmin([abs(pt[1] - yc) for yc in y_clusters])
    grid[y_idx, x_idx] = 0
    print(y_idx, x_idx, pt)

padding = 1  # 또는 2
grid_padded = cv2.copyMakeBorder(
    grid,
    top=padding,
    bottom=padding,
    left=padding,
    right=padding,
    borderType=cv2.BORDER_CONSTANT,
    value=255  # 흰색 padding
)

# 확대 후 저장
zoom = 10
grid_img = cv2.resize(grid_padded, (100, 180), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(r'd:\\TEST\\resultImage.png', grid_img)
cv2.imshow("test_generated_hough.png", grid_img)
cv2.waitKey(0)