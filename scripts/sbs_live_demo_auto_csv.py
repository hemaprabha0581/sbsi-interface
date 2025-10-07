import os
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa
import joblib
import winsound
import csv
from datetime import datetime

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 2  # seconds
DATA_FOLDER = r"C:\Medispeak\data"
MODEL_PATH = r"C:\Medispeak\scripts\sbs_model.pkl"
CSV_FILE = os.path.join(DATA_FOLDER, "live_demo_metadata.csv")

# ---------------- CLASSES ----------------
CLASSES = ["Deep_Breath", "Fast_Breath", "Whisper_Hello", "Whisper_Stop", "Cough", "Silence"]

# ---------------- LOAD MODEL ----------------
clf, le = joblib.load(MODEL_PATH)

# ---------------- HELPERS ----------------
def beep_sound():
    duration = 200  # milliseconds
    freq = 1000
    winsound.Beep(freq, duration)

def extract_features(samples):
    y = samples.astype(float)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE))
    return rms, zcr, spectral_centroid, spectral_bandwidth

def normalize_audio(samples, target_peak=10000):
    current_peak = np.max(np.abs(samples))
    if current_peak == 0:
        return samples
    factor = target_peak / current_peak
    return np.clip((samples * factor), -32768, 32767).astype(np.int16)

def record_sample(class_name, sample_num):
    filename = f"{class_name}_{sample_num:03d}.wav"
    filepath = os.path.join(DATA_FOLDER, filename)
    
    print(f"\n🎯 Recording {class_name} Sample #{sample_num}")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    beep_sound()
    print("🔴 Recording now!")
    samples = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    samples = samples.flatten()
    samples = normalize_audio(samples)
    wavfile.write(filepath, SAMPLE_RATE, samples)
    print(f"✅ Sample saved: {filepath}")
    return samples, filepath

def log_to_csv(filename, class_name, sample_num, rms, zcr, spectral_centroid, spectral_bandwidth):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Filename", "Class", "Sample_Num", "RMS", "ZCR", "Spectral_Centroid", "Spectral_Bandwidth", "Timestamp"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Filename": filename,
            "Class": class_name,
            "Sample_Num": sample_num,
            "RMS": rms,
            "ZCR": zcr,
            "Spectral_Centroid": spectral_centroid,
            "Spectral_Bandwidth": spectral_bandwidth,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# ---------------- LIVE AUTO DEMO ----------------
if __name__ == "__main__":
    print("🚀 Automatic Live Demo Recording & Prediction with CSV Logging")
    input("Press Enter to start recording all classes sequentially...")

    for class_name in CLASSES:
        for sample_num in range(1, 3):  # adjust number of samples per class
            samples, path = record_sample(class_name, sample_num)
            rms, zcr, spectral_centroid, spectral_bandwidth = extract_features(samples)
            pred = clf.predict([[rms, zcr, spectral_centroid, spectral_bandwidth]])
            class_pred = le.inverse_transform(pred)[0]
            print(f"🎉 Predicted Class: {class_pred}")
            
            log_to_csv(os.path.basename(path), class_name, sample_num, rms, zcr, spectral_centroid, spectral_bandwidth)
            time.sleep(1)  # short pause between samples

    print("\n✅ All samples recorded, predicted, and metadata logged to CSV!")



