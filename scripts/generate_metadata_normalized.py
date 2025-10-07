# generate_metadata_normalized.py
import os
import wave
import numpy as np
import csv
import time

# ---------------- CONFIG ----------------
DATA_FOLDER = r"C:\Medispeak\data"
CSV_PATH = os.path.join(DATA_FOLDER, "metadata.csv")
DEVICE_NAME = "Airdopes Alpha"
DEVICE_INDEX = 1
DISTANCE_CM = 8

# ---------------- RECORDING INSTRUCTIONS ----------------
instructions = {
    "Deep_Breath": [
        "Slow, steady inhale", "Slow, steady inhale", "Slow, steady inhale", "Slow, steady inhale", "Slow, steady inhale",
        "Slightly faster inhale","Slightly faster inhale","Slightly faster inhale","Slightly faster inhale","Slightly faster inhale",
        "Deep inhale + small exhale","Deep inhale + small exhale","Deep inhale + small exhale","Deep inhale + small exhale","Deep inhale + small exhale",
        "Deep inhale (hold breath 1s)","Deep inhale (hold breath 1s)","Deep inhale (hold breath 1s)","Deep inhale (hold breath 1s)","Deep inhale (hold breath 1s)"
    ],
    "Fast_Breath": [
        "Short, quick breaths","Short, quick breaths","Short, quick breaths","Short, quick breaths","Short, quick breaths",
        "Rapid in/out breaths","Rapid in/out breaths","Rapid in/out breaths","Rapid in/out breaths","Rapid in/out breaths",
        "Slightly longer exhales","Slightly longer exhales","Slightly longer exhales","Slightly longer exhales","Slightly longer exhales",
        "3 very quick breaths","3 very quick breaths","3 very quick breaths","3 very quick breaths","3 very quick breaths"
    ],
    "Whisper_Hello": [
        "Whisper 'hello' softly","Whisper 'hello' softly","Whisper 'hello' softly","Whisper 'hello' softly","Whisper 'hello' softly",
        "Whisper slightly faster","Whisper slightly faster","Whisper slightly faster","Whisper slightly faster","Whisper slightly faster",
        "Whisper 'hello' twice","Whisper 'hello' twice","Whisper 'hello' twice","Whisper 'hello' twice","Whisper 'hello' twice",
        "Whisper 'hello' with pause","Whisper 'hello' with pause","Whisper 'hello' with pause","Whisper 'hello' with pause","Whisper 'hello' with pause"
    ],
    "Whisper_Stop": [
        "Whisper 'stop' softly","Whisper 'stop' softly","Whisper 'stop' softly","Whisper 'stop' softly","Whisper 'stop' softly",
        "Whisper 'stop' longer","Whisper 'stop' longer","Whisper 'stop' longer","Whisper 'stop' longer","Whisper 'stop' longer",
        "Whisper 'stop' twice","Whisper 'stop' twice","Whisper 'stop' twice","Whisper 'stop' twice","Whisper 'stop' twice",
        "Whisper 'stop' slowly","Whisper 'stop' slowly","Whisper 'stop' slowly","Whisper 'stop' slowly","Whisper 'stop' slowly"
    ],
    "Cough": [
        "Single soft cough","Single soft cough","Single soft cough","Single soft cough","Single soft cough",
        "Two short coughs","Two short coughs","Two short coughs","Two short coughs","Two short coughs",
        "Single loud cough","Single loud cough","Single loud cough","Single loud cough","Single loud cough",
        "Short cough bursts","Short cough bursts","Short cough bursts","Short cough bursts","Short cough bursts"
    ],
    "Silence": ["Stay silent"]*20,
    "Short_Breath": [
        "Short, single inhales"]*10 + ["Two short inhales per sample"]*10,
    "Long_Breath": [
        "Long inhale (2s)"]*10 + ["Inhale + exhale (4s)"]*10
}

# ---------------- HELPER FUNCTIONS ----------------
def rms_peak(samples):
    samples_float = samples.astype(float)
    rms_val = np.sqrt(np.mean(samples_float**2))
    peak_val = np.max(np.abs(samples_float))
    return round(rms_val,1), int(peak_val)

def normalize_audio(samples, target_peak=10000):
    current_peak = np.max(np.abs(samples))
    if current_peak == 0:
        return samples
    factor = target_peak / current_peak
    return np.clip((samples * factor), -32768, 32767).astype(np.int16)

def append_csv(row):
    header = ["filename","class","sample_no","rms","peak_amp","instruction","timestamp","device_name","device_index","distance_cm","notes"]
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
        
        # Read WAV
        with wave.open(file_path, 'rb') as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        
        # Normalize
        samples_norm = normalize_audio(samples)
        
        # Save normalized back
        with wave.open(file_path, 'wb') as wf:
            wf.setparams(params)
            wf.writeframes(samples_norm.tobytes())
        
        # RMS & peak after normalization
        rms_val, peak_val = rms_peak(samples_norm)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        sample_no = i+1
        note = "ok"
        instruction = instructions.get(class_name, [""]*len(files))[i] if class_name in instructions else ""
        
        row = [file,class_name,sample_no,rms_val,peak_val,instruction,timestamp,DEVICE_NAME,DEVICE_INDEX,DISTANCE_CM,note]
        append_csv(row)
        print(f"Processed: {file} | RMS: {rms_val} | Peak: {peak_val}")

print("\n✅ metadata.csv with normalized audio & instructions generated successfully!")
