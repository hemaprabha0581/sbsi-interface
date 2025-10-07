from flask import Flask, render_template, request, jsonify
import joblib
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained model
model_path = r"C:\Medispeak\scripts\sbs_model.pkl"
clf, le = joblib.load(model_path)

UPLOAD_FOLDER = r"C:\Medispeak\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No file received"})

    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        y, sr = librosa.load(filepath, sr=None)
        features = [
            np.mean(librosa.feature.rms(y=y)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        ]
        X = np.array([features])
        pred = clf.predict(X)[0]
        label = le.inverse_transform([pred])[0]
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
