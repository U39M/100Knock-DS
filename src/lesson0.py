import os
import pandas as pd
os.chdir('../csv/Part1/')

file = 'height_data.csv'
# ファイルの読み込み
df = pd.read_csv(file)

# 上から5行を表示
print(df.head())