# C:\Medispeak\scripts\generate_metadata.py

import os
import wave
import numpy as np
import csv
import time

# ---------------- CONFIG ----------------
DATA_FOLDER = r"C:\Medispeak\data"
CSV_PATH = os.path.join(DATA_FOLDER, "metadata.csv")
DEVICE_NAME = "Airdopes Alpha"   # change if needed
DEVICE_INDEX = 1                 # change if needed
DISTANCE_CM = 8                  # optional distance from mouth

# ---------------- HELPER FUNCTIONS ----------------
def rms_peak(audio_bytes):
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(float)
    rms_val = np.sqrt(np.mean(samples**2))
    peak_val = np.max(np.abs(samples))
    return round(rms_val, 1), int(peak_val)

def append_csv(row):
    header = ["filename","class","sample_no","rms","peak_amp","timestamp","device_name","device_index","distance_cm","notes"]
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH,"w",newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(CSV_PATH,"a",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ---------------- MAIN ----------------
classes = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER,d))]

for class_name in classes:
    class_path = os.path.join(DATA_FOLDER, class_name)
    files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(".wav")])
    
    for i, file in enumerate(files):
        file_path = os.path.join(class_path, file)
        
        # Read WAV bytes
        with wave.open(file_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
        
        rms, peak = rms_peak(frames)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        sample_no = i+1
        notes = "ok"
        
        row = [file,class_name,sample_no,rms,peak,timestamp,DEVICE_NAME,DEVICE_INDEX,DISTANCE_CM,notes]
        append_csv(row)
        print(f"Added to CSV: {file}")

print("\n✅ metadata.csv generated in your data folder!")
