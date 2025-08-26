import os

def change_extension(directory, old_ext, new_ext):
    for filename in os.listdir(directory):
        if filename.endswith(old_ext):
            base = os.path.splitext(filename)[0]
            new_filename = base + new_ext
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"{filename} → {new_filename}")

# 사용 예시
change_extension("d:/Non_Documents", ".py", ".txt")