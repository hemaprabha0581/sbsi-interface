# medispeak/scripts/record_medispeak_pyaudio.py
import pyaudio
import wave
import numpy as np
import os
import csv
import time
import winsound

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
RMS_MIN = 100     # raw amplitude for check
RMS_MAX = 3000
PEAK_MIN = 200
PEAK_MAX = 30000
OUT_DIR = "../data"
CSV_PATH = os.path.join(OUT_DIR, "metadata.csv")

# ---------------- HELPER FUNCTIONS ----------------
def rms(data):
    samples = np.frombuffer(data, dtype=np.int16)
    return np.sqrt(np.mean(samples**2))

def peak(data):
    samples = np.frombuffer(data, dtype=np.int16)
    return np.max(np.abs(samples))

def beep():
    frequency = 1000
    duration_ms = 200
    winsound.Beep(frequency, duration_ms)

def countdown(seconds=3):
    print("Get ready to record:")
    for i in range(seconds,0,-1):
        print(f"{i}...")
        time.sleep(1)
    print("Recording now!")

def append_csv(row):
    header = ["filename","class","sample_no","instruction","rms","peak_amp","timestamp","device","distance_cm","notes"]
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH,"w",newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(CSV_PATH,"a",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def record_sample(duration_sec=2):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK * duration_sec)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)

# ---------------- INSTRUCTIONS TABLE ----------------
instructions_table = {
    "Deep_Breath": ["Slow, steady inhale"]*5 + ["Slightly faster inhale"]*5 + ["Deep inhale + small exhale"]*5 + ["Deep inhale (hold breath 1s)"]*5,
    "Fast_Breath": ["Short, quick breaths"]*5 + ["Rapid in/out breaths"]*5 + ["Slightly longer exhales"]*5 + ["3 very quick breaths"]*5,
    "Whisper_Hello": ["Whisper 'hello' softly"]*5 + ["Whisper slightly faster"]*5 + ["Whisper 'hello' twice"]*5 + ["Whisper 'hello' with pause"]*5,
    "Whisper_Stop": ["Whisper 'stop' softly"]*5 + ["Whisper 'stop' longer"]*5 + ["Whisper 'stop' twice"]*5 + ["Whisper 'stop' slowly"]*5,
    "Cough": ["Single soft cough"]*5 + ["Two short coughs"]*5 + ["Single loud cough"]*5 + ["Short cough bursts"]*5,
    "Silence": ["Stay silent"]*20,
    "Short_Breath": ["Short, single inhales"]*10 + ["Two short inhales per sample"]*10,
    "Long_Breath": ["Long inhale (2s)"]*10 + ["Inhale + exhale (4s)"]*10
}

# ---------------- MAIN RECORDING ----------------
def main():
    print("Welcome to Medispeak recording script using PyAudio!")
    device = input("Device name (e.g., AirdopesAlpha): ").strip() or "AirdopesAlpha"
    distance = input("Distance from mouth in cm (e.g., 8): ").strip() or "8"

    classes = list(instructions_table.keys())
    
    for class_name in classes:
        folder_path = os.path.join(OUT_DIR, class_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder '{folder_path}' does not exist! Please create it.")
            continue
        
        print(f"\n➡️ Starting class: {class_name}")
        num_samples = len(instructions_table[class_name])
        
        for i in range(num_samples):
            sample_no = i+1
            instruction = instructions_table[class_name][i]
            print(f"\nClass: {class_name} | Sample {sample_no}/{num_samples}")
            print(f"Instruction: {instruction}")
            input("Press ENTER when ready...")
            
            countdown(3)
            beep()
            
            duration = 2
            if "4s" in instruction:
                duration = 4
            elif "2s" in instruction:
                duration = 2
            
            while True:
                audio_data = record_sample(duration)
                r = rms(audio_data)
                p_amp = peak(audio_data)
                print(f"RMS={r:.1f}, Peak={p_amp}")
                
                note = "ok"
                action = ""
                if r < RMS_MIN:
                    print("⚠️ Audio too soft")
                    action = input("Press (r) to re-record, (k) to keep anyway, (s) to skip: ").strip().lower()
                elif r > RMS_MAX:
                    print("⚠️ Audio too loud")
                    action = input("Press (r) to re-record, (k) to keep anyway, (s) to skip: ").strip().lower()
                elif p_amp < PEAK_MIN or p_amp > PEAK_MAX:
                    print("⚠️ Peak amplitude out of range")
                    action = input("Press (r) to re-record, (k) to keep anyway, (s) to skip: ").strip().lower()
                
                if action=="r" or action=="":
                    continue
                elif action=="k":
                    note="kept_out_of_range"
                elif action=="s":
                    note="skipped"
                    break
                
                if note != "skipped":
                    fname = f"{class_name}_{sample_no:03d}.wav"
                    file_path = os.path.join(folder_path,fname)
                    wf = wave.open(file_path, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_data)
                    wf.close()
                    
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    row = [fname,class_name,sample_no,instruction,f"{r:.1f}",f"{p_amp}",timestamp,device,distance,note]
                    append_csv(row)
                    print(f"Saved {fname} ✅")
                break
    
    print("\n🎉 All classes recorded! Check your data folder and metadata.csv")

if __name__=="__main__":
    main()

