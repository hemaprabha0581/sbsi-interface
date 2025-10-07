import torch
import torch.nn as nn
from pathlib import Path

# Create a simple model: 384 -> 64 -> 2
FEATURE_SIZE = 384
model = nn.Sequential(
    nn.Linear(FEATURE_SIZE, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 2),
)

# Trace with an example input to get a TorchScript module
example = torch.randn(1, FEATURE_SIZE)
scripted = torch.jit.trace(model, example)

# Save to backend/model.pt relative to this script
backend_dir = Path(__file__).resolve().parents[1]
out_path = backend_dir / "model.pt"
scripted.save(str(out_path))
print(f"Saved TorchScript model to {out_path}")
