import io
import wave
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False

logger = logging.getLogger("sbsi")
logging.basicConfig(level=logging.INFO)

LABELS = ["breath", "speech"]
FEATURE_SIZE = 384  # 128 freq bins * (mean, std, max)


class DummyModel(nn.Module):
    def __init__(self, in_features: int = FEATURE_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model() -> nn.Module:
    """Load a PyTorch model from backend/model.pt. If missing, use DummyModel."""
    # backend/app/main.py -> backend directory is parents[1]
    backend_dir = Path(__file__).resolve().parents[1]
    model_path = backend_dir / "model.pt"

    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        try:
            # Try TorchScript first
            model = torch.jit.load(str(model_path), map_location="cpu")
            model.eval()
            logger.info("Loaded TorchScript model.")
            return model
        except Exception as e_ts:
            logger.warning(f"TorchScript load failed: {e_ts}. Trying torch.load...")
            try:
                model = torch.load(str(model_path), map_location="cpu")
                if hasattr(model, "eval"):
                    model.eval()
                logger.info("Loaded torch.save() model.")
                return model
            except Exception as e_pt:
                logger.error(f"Failed to load model: {e_pt}. Falling back to DummyModel.")
    else:
        logger.warning(f"Model file not found at {model_path}. Using DummyModel.")

    dummy = DummyModel(FEATURE_SIZE)
    dummy.eval()
    return dummy


def read_wav_bytes(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Read WAV from bytes to mono float32 numpy array in [-1, 1] and return (audio, sr).
    Supports 8-bit (unsigned), 16-bit, and 32-bit float PCM WAV. Stereo is converted to mono.
    """
    with wave.open(io.BytesIO(file_bytes), "rb") as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    if sampwidth == 1:
        # 8-bit unsigned PCM
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        # 16-bit PCM
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        data = data / 32768.0
    elif sampwidth == 4:
        # Could be 32-bit PCM or 32-bit float; assume float32 PCM
        # If int32 PCM, the scale would differ, but this is uncommon for simple recordings.
        data = np.frombuffer(frames, dtype=np.float32)
        # If it's actually int32, you can adapt by viewing and scaling accordingly.
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)  # convert to mono

    return data.astype(np.float32), int(fr)


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract a simple log-spectrogram-based feature vector of fixed size (384)."""
    if audio.size == 0:
        raise ValueError("Empty audio")

    # Optional normalization
    max_abs = max(1e-8, float(np.max(np.abs(audio))))
    audio = (audio / max_abs).astype(np.float32)

    # Compute spectrogram
    nperseg = 512
    noverlap = 256

    if SCIPY_AVAILABLE:
        freqs, times, Sxx = signal.spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
        spec = np.log1p(Sxx).astype(np.float32)  # (freq, time)
    else:
        # Minimal STFT implementation
        hop = nperseg - noverlap
        if audio.shape[0] < nperseg:
            pad = nperseg - audio.shape[0]
            audio = np.pad(audio, (0, pad))
        num_frames = 1 + (audio.shape[0] - nperseg) // hop
        window = np.hanning(nperseg).astype(np.float32)
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(num_frames, nperseg),
            strides=(audio.strides[0] * hop, audio.strides[0]),
            writeable=False,
        )
        frames = frames * window[None, :]
        spec_mag = np.abs(np.fft.rfft(frames, axis=1)).T  # (freq, time)
        spec = np.log1p(spec_mag).astype(np.float32)

    # Limit to first 128 freq bins for stability
    freq_bins = min(128, spec.shape[0])
    spec = spec[:freq_bins, :]

    # Aggregate over time: mean, std, max per freq bin -> 3 * 128 = 384
    mean_feat = spec.mean(axis=1)
    std_feat = spec.std(axis=1)
    max_feat = spec.max(axis=1)
    feat = np.concatenate([mean_feat, std_feat, max_feat], axis=0)

    # Ensure fixed size (pad or truncate)
    if feat.shape[0] < FEATURE_SIZE:
        feat = np.pad(feat, (0, FEATURE_SIZE - feat.shape[0]))
    else:
        feat = feat[:FEATURE_SIZE]

    return feat.astype(np.float32)


app = FastAPI(title="SBSI Backend", version="0.1.0")

# CORS for local frontend
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        audio, sr = read_wav_bytes(file_bytes)
        features = extract_features(audio, sr)
        x = torch.from_numpy(features).unsqueeze(0)  # (1, FEATURE_SIZE)

        with torch.no_grad():
            logits = MODEL(x.float())
            if not isinstance(logits, torch.Tensor):  # In case model returns tuple
                logits = torch.as_tensor(logits)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = LABELS[pred_idx] if pred_idx < len(LABELS) else str(pred_idx)
        confidence = float(probs[pred_idx])

        return {
            "predicted_class": pred_label,
            "confidence": confidence,
        }
    except wave.Error as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
