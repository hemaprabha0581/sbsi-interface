import os
import csv
import random

# ---------------- CONFIG ----------------
DATA_FOLDER = r"C:\Medispeak\data"
INPUT_CSV = os.path.join(DATA_FOLDER, "metadata.csv")
OUTPUT_CSV = os.path.join(DATA_FOLDER, "metadata_train_test.csv")
TRAIN_RATIO = 0.8

# ---------------- READ METADATA ----------------
metadata_rows = []
with open(INPUT_CSV, 'r', newline='') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    for row in reader:
        # Keep only keys that exist in headers
        clean_row = {k: row[k] for k in headers}
        metadata_rows.append(clean_row)

# ---------------- ADD 'set' COLUMN ----------------
headers.append('set')

# ---------------- SPLIT TRAIN/TEST ----------------
classes = set([row['class'] for row in metadata_rows])
split_rows = []

for cls in classes:
    cls_rows = [row for row in metadata_rows if row['class'] == cls]
    random.shuffle(cls_rows)
    train_count = int(len(cls_rows) * TRAIN_RATIO)
    
    for i, row in enumerate(cls_rows):
        row['set'] = 'train' if i < train_count else 'test'
        split_rows.append(row)

# ---------------- WRITE NEW CSV ----------------
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for row in split_rows:
        writer.writerow(row)

print(f"\n✅ metadata_train_test.csv generated successfully with train/test split!")
