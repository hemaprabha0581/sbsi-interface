from flask import Flask, render_template, jsonify
import sounddevice as sd
import numpy as np
import librosa
import joblib
import scipy.io.wavfile as wavfile
import os
import time

app = Flask(__name__)

MODEL_PATH = r"C:\Medispeak\scripts\sbs_model.pkl"
clf, le = joblib.load(MODEL_PATH)
SAMPLE_RATE = 16000
DURATION = 2  # seconds

def record_audio():
    print("🎤 Recording started...")
    samples = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    samples = samples.flatten()
    print("✅ Recording finished.")
    return samples

def extract_features(samples):
    y = samples.astype(float)
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE))
    return [rms, zcr, spectral_centroid, spectral_bandwidth]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_live():
    samples = record_audio()
    features = extract_features(samples)
    pred = clf.predict([features])
    class_name = le.inverse_transform(pred)[0]
    print(f"🎯 Predicted Class: {class_name}")
    return jsonify({'prediction': class_name})

if __name__ == "__main__":
    app.run(debug=True)
