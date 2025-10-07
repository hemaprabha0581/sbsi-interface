import math, wave, struct
sr = 16000
freq = 440.0
seconds = 1.0
n = int(sr * seconds)
path = r"C:\\Medispeak\\sample.wav"
with wave.open(path, 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    for i in range(n):
        s = 0.2 * math.sin(2 * math.pi * freq * i / sr)
        w.writeframes(struct.pack('<h', int(max(-1.0, min(1.0, s)) * 32767)))
print(f"WAV written: {path}")
