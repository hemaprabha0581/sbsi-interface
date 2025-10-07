import io
import math
import struct
import wave
from starlette.testclient import TestClient
from app.main import app

def make_wav_bytes(sr=16000, freq=440.0, seconds=0.5, amp=0.2):
    n = int(sr * seconds)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            s = amp * math.sin(2 * math.pi * freq * i / sr)
            w.writeframes(struct.pack('<h', int(max(-1.0, min(1.0, s)) * 32767)))
    return buf.getvalue()

def test_health_ok():
    client = TestClient(app)
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_with_sine_wav():
    client = TestClient(app)
    audio_bytes = make_wav_bytes()
    files = {
        "file": ("test.wav", audio_bytes, "audio/wav"),
    }
    r = client.post('/predict', files=files)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["breath", "speech"]
    assert "confidence" in data
    assert 0.0 <= float(data["confidence"]) <= 1.0
