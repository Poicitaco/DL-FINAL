"""
Gop data tu nhieu nguoi thu thanh 1 file.
Dat cac file CSV vao data/raw/ roi chay script nay.
"""
import pandas as pd
import glob
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUTPUT  = os.path.join(RAW_DIR, 'gesture_data.csv')

# Tim tat ca file CSV trong raw/ (tru file tong hop)
files = [f for f in glob.glob(os.path.join(RAW_DIR, '*.csv'))
         if os.path.basename(f) != 'gesture_data.csv']

if not files:
    print("Khong tim thay file CSV nao trong data/raw/")
    print("Dat cac file gesture_data_nguoi1.csv, gesture_data_nguoi2.csv... vao day")
    exit()

print(f"Tim thay {len(files)} file:")
dfs = []
for f in sorted(files):
    df = pd.read_csv(f)
    print(f"  {os.path.basename(f)}: {len(df)} dong, person_id={df['person_id'].unique() if 'person_id' in df.columns else 'N/A'}")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(OUTPUT, index=False)

print(f"\nDa gop: {len(merged)} dong tong cong -> {OUTPUT}")
print(f"Phan bo person_id:")
if 'person_id' in merged.columns:
    print(merged['person_id'].value_counts().to_string())
