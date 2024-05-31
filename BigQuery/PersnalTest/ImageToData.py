import cv2
import pytesseract as pt
import pandas as pd
import re

pt.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
image_path = 'D:\Tesseract-OCR\TestImage.png'
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

custom_config = r'--oem 3 --psm 6'
text = pt.image_to_string(gray, config=custom_config)

print("Extract Text : ")
print(text)

date_pattern = r'\d{4}-\d{2}-\d{2}' 
x_values = re.findall(date_pattern, text)

df = pd.DataFrame(x_values, columns=['X-axis Values'])

print(x_values)

output_excel_path = 'x_axis_values.xlsx'
df.to_excel(output_excel_path, index=False)