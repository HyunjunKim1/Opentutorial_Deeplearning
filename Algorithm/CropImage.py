
import os
from PIL import Image

def crop_images_center(input_root, output_root):
    """
    입력 폴더 내 모든 .png 이미지를 300x300으로 중앙 기준 자르기
    폴더 구조는 그대로 출력 폴더에 유지
    """
    # 출력 폴더가 없으면 생성
    os.makedirs(output_root, exist_ok=True)

    # 입력 폴더 내 모든 파일 및 하위 폴더를 탐색
    for dirpath, dirnames, filenames in os.walk(input_root):
        # 출력 폴더 내 상대 경로 계산
        rel_path = os.path.relpath(dirpath, input_root)
        output_dir = os.path.join(output_root, rel_path)
        os.makedirs(output_dir, exist_ok=True)  # 출력 폴더 생성

        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(output_dir, filename)

                try:
                    with Image.open(input_path) as img:
                        # 500x500을 가정 (아니면 경고 출력)
                        if img.width != 500 or img.height != 500:
                            print(f"⚠ 경고: {input_path}는 500x500이 아님 (현재: {img.size})")
                            print("     계속 진행되지만, 중앙 기준으로 300x300 자릅니다.")

                        # 중앙 좌표 계산
                        left = (img.width - 300) // 2
                        top = (img.height - 300) // 2
                        right = left + 300
                        bottom = top + 300

                        cropped_img = img.crop((left, top, right, bottom))
                        cropped_img.save(output_path)
                        print(f"✅ 저장됨: {output_path}")

                except Exception as e:
                    print(f"❌ 오류 발생: {input_path} - {e}")

if __name__ == "__main__":
    print("🖼️ 이미지 중앙 Crop 프로그램 시작")
    input_root = input("입력 폴더 경로를 입력하세요: ").strip().strip('"')
    output_root = input("출력 폴더 경로를 입력하세요: ").strip().strip('"')

    if not os.path.isdir(input_root):
        print(f"❌ 입력 폴더가 존재하지 않습니다: {input_root}")
    else:
        crop_images_center(input_root, output_root)
        print("🎉 모든 이미지 처리 완료!")