import os
import cv2
import numpy as np
import keyboard  # pip install keyboard

def simulate_inline_capture(
    image_path: str,
    output_dir: str = "captures",
    pixel_pitch_mm: float = 0.014,
    fov_mm: tuple[float, float] = (120.0, 90.0),
    overlap_mm: float = 20.0,
    line_speed_mpm: float = 10.0,
) -> None:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    H, W = img.shape[:2]

    fov_h_px = int(round(fov_mm[1] / pixel_pitch_mm))      # 세로 FOV
    step_mm  = fov_mm[1] - overlap_mm
    step_px  = int(round(step_mm / pixel_pitch_mm))

    line_speed_mm_per_sec = line_speed_mpm * 1000 / 60.0
    capture_interval_sec = step_mm / line_speed_mm_per_sec

    os.makedirs(output_dir, exist_ok=True)

    y = 0
    idx = 0
    current_time = 0.0
    prev_file = None
    print(f"▶ 라인 속도: {line_speed_mpm} m/min ({line_speed_mm_per_sec:.2f} mm/s)")
    print(f"▶ 촬영 간격: {capture_interval_sec:.2f}초 / 이동 거리: {step_mm}mm\n")

    top_reused = False
    top_saved = None

    while not keyboard.is_pressed('0'):
        if y + fov_h_px <= H:
            # 일반 컷
            tile = img[y : y + fov_h_px, :]
        else:
            # 마지막 Merge 컷
            bottom = img[y:H, :]
            top_needed = fov_h_px - bottom.shape[0]
            top = img[0:top_needed, :]
            tile = np.vstack([bottom, top])
            top_saved = top  # top 저장
            top_reused = True

        idx += 1
        fname = os.path.join(output_dir, f"frame_00.bmp")
        cv2.imwrite(fname, tile)
        print(f"[{current_time:6.2f} sec] frame_00 촬영 (y={y}px)")

        # 다음 위치 설정
        y += step_px
        current_time += capture_interval_sec

        # Merge 컷 이후 루프 재시작 조건
        if top_reused and y >= H:
            y = top_needed  # top 다음 줄부터 다시 시작
            top_reused = False
            print(f"🔁 Merge 이후 루프 재시작 → y = {y}px")

    print(f"\n'0' 입력 감지 → 시뮬레이션 종료. 총 촬영 컷 수: {idx}, 시간: {current_time:.2f}초")

# 실행
if __name__ == "__main__":
    simulate_inline_capture(
        image_path = r"E:\LF\LF_GrabImage\GrabImage_00.bmp",
        output_dir = r"E:\LF\LF_GrabImage\captures",
        pixel_pitch_mm = 0.014,
        fov_mm = (120.0, 90.0),
        overlap_mm = 20.0,
        line_speed_mpm = 5.0
    )
