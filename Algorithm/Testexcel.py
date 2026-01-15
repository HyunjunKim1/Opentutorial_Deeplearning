import pandas as pd

file_name = r"d:\personal_space\평가용데이터\상부\Flash\Lead\Defect Data (Flash_Lead).xlsx"

df = pd.read_excel(file_name)

print(df)