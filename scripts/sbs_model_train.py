import os
import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------
DATA_FOLDER = r"C:\Medispeak\data"
CSV_PATH = os.path.join(DATA_FOLDER, "metadata_train_test.csv")
SAMPLE_RATE = 16000  # make sure it matches your recordings

# ---------------- LOAD CSV ----------------
df = pd.read_csv(CSV_PATH)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    # Basic features
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    return [rms, zcr, spectral_centroid, spectral_bandwidth]

features = []
labels = []
sets = []

for idx, row in df.iterrows():
    file_path = os.path.join(DATA_FOLDER, row['class'], row['filename'])
    if not os.path.exists(file_path):
        continue
    feat = extract_features(file_path)
    features.append(feat)
    labels.append(row['class'])
    sets.append(row['set'])

features = np.array(features)
labels = np.array(labels)
sets = np.array(sets)

# ---------------- ENCODE LABELS ----------------
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# ---------------- SPLIT TRAIN/TEST ----------------
X_train = features[sets == 'train']
y_train = labels_encoded[sets == 'train']
X_test = features[sets == 'test']
y_test = labels_encoded[sets == 'test']

# ---------------- TRAIN MODEL ----------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ---------------- TEST MODEL ----------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy on test set: {acc*100:.2f}%")

# ---------------- PREDICT NEW SAMPLE FUNCTION ----------------
def predict_new(file_path):
    feat = extract_features(file_path)
    pred = clf.predict([feat])
    class_name = le.inverse_transform(pred)[0]
    return class_name

# Example usage:
# new_sample = r"C:\Medispeak\data\Deep_Breath\Deep_Breath_001.wav"
# print("Predicted Class:", predict_new(new_sample))
