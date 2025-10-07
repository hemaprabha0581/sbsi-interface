import os
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa
import joblib
import simpleaudio as sa  # for beep sound

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 2  # seconds
DATA_FOLDER = r"C:\Medispeak\data"
MODEL_PATH = r"C:\Medispeak\scripts\sbs_model.pkl"  # your trained model

# ---------------- LOAD MODEL ----------------
clf, le = joblib.load(MODEL_PATH)

# ---------------- HELPERS ----------------
def beep_sound():
    frequency = 1000  # Hz
    fs = 44100
    t = np.linspace(0, 0.2, int(fs*0.2), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = (tone * 32767).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

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
    print("\n🎯 Get ready! Countdown starting...")
    for i in range(3,0,-1):
        print(f"{i}...")
        time.sleep(1)
    beep_sound()
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
