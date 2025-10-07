import os
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa
import joblib  # for saving/loading model
from sklearn.preprocessing import LabelEncoder

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 2  # seconds
DATA_FOLDER = r"C:\Medispeak\data"
MODEL_PATH = r"C:\Medispeak\scripts\sbs_model.pkl"  # Save your trained model here

# ---------------- LOAD MODEL ----------------
clf, le = joblib.load(MODEL_PATH)

# ---------------- HELPER FUNCTIONS ----------------
def extract_features(samples):
    y = samples.astype(float)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE))
    return [rms, zcr, spectral_centroid, spectral_bandwidth]

def normalize_audio(samples, target_peak=10000):
    current_peak = np.max(np.abs(samples))
    if current_peak == 0:
        return samples
    factor = target_peak / current_peak
    return np.clip((samples * factor), -32768, 32767).astype(np.int16)

def record_sample(filename):
    print("\n🎯 Get ready to record! Counting down...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("🔴 Recording now!")
    samples = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    samples = samples.flatten()
    samples = normalize_audio(samples)
    wavfile.write(filename, SAMPLE_RATE, samples)
    print(f"✅ Sample saved: {filename}")
    return samples

# ---------------- LIVE DEMO ----------------
if __name__ == "__main__":
    sample_name = input("Enter filename for new sample (e.g., demo_001.wav): ")
    sample_path = os.path.join(DATA_FOLDER, sample_name)
    
    samples = record_sample(sample_path)
    features = extract_features(samples)
    pred = clf.predict([features])
    class_name = le.inverse_transform(pred)[0]
    print(f"\n🎉 Predicted Class: {class_name}")
