
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. 경로 설정 (raw-string 사용 → \UXXXX 오류 방지)
# ----------------------------------------------------------------------
EXCEL_PATH = Path(r"D:\Git\ResultCompare\ResultCompare\bin\x64\Debug\DefectData.xlsx")
IMAGE_FOLDERS = [
    Path(r"C:\Users\admin\Desktop\TestData\상부"),
    Path(r"C:\Users\admin\Desktop\TestData\하부"),
    Path(r"C:\Users\admin\Desktop\TestData\투과"),
]
RESULT_PATH = Path(r"D:\ResultData.xlsx")  # 출력 파일

# ----------------------------------------------------------------------
# 2. Excel 읽기 (헤더 없음 → 컬럼을 0,1,2,3,4 로 간주)
# ----------------------------------------------------------------------
def load_defect_data(excel_path: Path) -> pd.DataFrame:
    """헤더 없이 Excel 읽기 → 컬럼명: 0,1,2,3,4"""
    if not excel_path.is_file():
        raise FileNotFoundError(f"엑셀 파일 없음: {excel_path}")

    # header=None → 첫 행도 데이터로 취급
    df = pd.read_excel(excel_path, engine="openpyxl", header=None)

    if df.shape[1] < 5:
        raise ValueError(f"엑셀 파일에 최소 5열이 필요합니다. 현재: {df.shape[1]}열")

    # 컬럼명을 0,1,2,3,4 로 유지 (A,B,C,D,E 대신 인덱스 사용)
    # 4번째 열 = 인덱스 3
    df["file_col"] = df[3].astype(str).str.strip().str.lower()  # 비교용 정규화
    return df

# ----------------------------------------------------------------------
# 3. 이미지 파일 수집
# ----------------------------------------------------------------------
def get_image_files(image_folders: list) -> list:
    """모든 이미지 폴더에서 이미지 파일 목록 수집"""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for folder in image_folders:
        if not folder.is_dir():
            print(f"[경고] 폴더 없음: {folder}")
            continue
        for ext in extensions:
            files.extend(folder.rglob(ext))
    return files

# ----------------------------------------------------------------------
# 4. 매칭: 4번째 열(인덱스 3) 값과 이미지 파일명(확장자 제외) 비교
# ----------------------------------------------------------------------
def find_matching_rows(df: pd.DataFrame, image_files: list) -> pd.DataFrame:
    """이미지 파일명과 4번째 열 값이 일치하는 행 추출"""
    matched = []
    for img_path in tqdm(image_files, desc="이미지 매칭 중", unit="file"):
        img_stem = img_path.stem.strip().lower()  # 확장자 제외, 소문자+공백제거

        # file_col 과 일치 여부 확인
        mask = df["file_col"] == img_stem
        if mask.any():
            # 원본 데이터의 0~4열(A~E에 해당)만 추출
            row = df.loc[mask, [0, 1, 2, 3, 4]].copy()
            matched.append(row)

    return pd.concat(matched, ignore_index=True) if matched else pd.DataFrame()

# ----------------------------------------------------------------------
# 5. 결과 저장 (기존 파일 있으면 이어 붙임)
# ----------------------------------------------------------------------
def save_result(final_df: pd.DataFrame, result_path: Path):
    """결과를 Excel로 저장. 기존 파일 있으면 합침."""
    if final_df.empty:
        print("❌ 매칭된 데이터가 없습니다.")
        # 빈 파일도 생성하지 않기 위해 종료
        return

    if result_path.is_file():
        existing_df = pd.read_excel(result_path, engine="openpyxl", header=None)
        combined_df = pd.concat([existing_df, final_df], ignore_index=True)
    else:
        combined_df = final_df

    # 저장 (헤더 없이)
    with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
        combined_df.to_excel(writer, sheet_name="Result", index=False, header=False)

    print(f"\n✅ 매칭된 {len(final_df)}개 행을 {result_path} 에 저장했습니다.")

# ----------------------------------------------------------------------
# 6. 메인
# ----------------------------------------------------------------------
def main():
    print("🚀 시작: 이미지 파일명과 4번째 열 매칭")

    # 1. 엑셀 로드 (헤더 없음)
    df_excel = load_defect_data(EXCEL_PATH)

    # 2. 이미지 파일 수집
    image_files = get_image_files(IMAGE_FOLDERS)
    if not image_files:
        print("❌ 이미지 파일이 하나도 없습니다.")
        return
    print(f"📁 {len(image_files)}개 이미지 검색됨")

    # 3. 매칭
    matched_df = find_matching_rows(df_excel, image_files)

    if matched_df.empty:
        print("❌ 일치하는 행이 없습니다.")
    else:
        print(f"✅ {len(matched_df)}개 일치")

    # 4. 저장
    save_result(matched_df, RESULT_PATH)

if __name__ == "__main__":
    main()