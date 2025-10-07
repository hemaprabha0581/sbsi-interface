# Silent Breath Speech Interface (SBSI)

A minimal full-stack web app scaffold to serve a PyTorch model via FastAPI and a React + Tailwind frontend.

Features
- FastAPI backend with /health and /predict endpoints
- File upload of audio and simple preprocessing placeholder (waveform normalization)
- PyTorch model loading from model.pt (TorchScript or torch.load())
- CORS enabled for frontend
- React + Vite + TailwindCSS frontend with audio uploader and predictor
- Dockerfiles for backend and frontend
- docker-compose to run both together

Project structure

```
C:\Medispeak
├── backend
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt
│   └── app
│       ├── __init__.py
│       ├── inference.py
│       ├── main.py
│       └── preprocessing.py
├── frontend
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── src
│       ├── App.jsx
│       ├── index.css
│       └── main.jsx
├── docker-compose.yml
└── .gitignore
```

Prerequisites
- Docker Desktop installed and running
- Your trained model saved as model.pt (TorchScript recommended) placed at backend/model.pt

Quick start with Docker
1) Place your model
- Copy your model file to backend/model.pt
- Optionally set CLASS_NAMES (comma-separated) if your model uses a specific label list.

2) Build and run
```powershell
# From C:\Medispeak
docker compose up -d --build
```
This will start:
- Backend at http://localhost:8000
- Frontend at http://localhost:5173

3) Test health endpoint
```powershell
curl http://localhost:8000/health
```
Expected: {"status":"ok"}

4) Open the app
- Navigate to http://localhost:5173
- Choose an audio file (WAV/MP3/etc.) and click Predict

Configuration
- Backend
  - MODEL_PATH: path to model file inside the container (default /app/model.pt)
  - CLASS_NAMES: comma-separated class labels (default: "breath,speech")
  - CORS_ORIGINS: comma-separated origins allowed (default: *)
- Frontend
  - VITE_API_BASE: API root used by the frontend (default: http://localhost:8000). For docker-compose, this is set at build-time in docker-compose.yml.

Model expectations and preprocessing
- The backend currently loads audio to a 16kHz mono waveform, normalizes amplitude, and passes the 1D float array to the model.
- If your model expects spectrograms or a different shape (e.g., [B, 1, T] or log-mel features), update:
  - backend/app/preprocessing.py (to compute MFCC/Mel/etc.)
  - backend/app/inference.py (to match the input tensor shape)
- CLASS_NAMES should match your model's output dimension ordering.

Local development without Docker (optional)
Backend (Python 3.11 recommended):
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MODEL_PATH = "C:\\Medispeak\\backend\\model.pt"
uvicorn app.main:app --reload --port 8000
```
Frontend:
```powershell
cd frontend
npm install
# Optional: copy .env.example to .env and customize VITE_API_BASE
npm run dev
```

Deploying the backend to Render
1) Push this repo to GitHub
2) On Render, create a new Web Service
- Repository: select this project
- Root Directory: backend
- Runtime: Python 3.x
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
- Environment Variables:
  - MODEL_PATH=/app/model.pt
  - CLASS_NAMES=breath,speech (or your label list)
  - CORS_ORIGINS=https://your-frontend-domain (add others as needed)
- Model file options:
  - Commit model.pt into backend/ (simple, but may be large), or
  - Download from object storage at startup (customize Dockerfile or a startup script).

Deploying the frontend to Vercel
1) Push this repo to GitHub
2) Import project in Vercel and set Root Directory to frontend
3) Framework Preset: Vite
4) Build Command: npm run build
5) Output Directory: dist
6) Environment Variables (Production):
   - VITE_API_BASE=https://your-render-backend.onrender.com
7) Redeploy; ensure your backend CORS allows the Vercel domain

Troubleshooting
- Audio decoding issues: ffmpeg and libsndfile are installed in the backend container. Ensure your audio format is supported. WAV is safest.
- Model load errors: If you saved a state_dict, you must load it into the model class and export a full model. TorchScript (torch.jit.trace/script) usually deploys more smoothly.
- CORS problems: Set CORS_ORIGINS on the backend to include your frontend origin.
- 404/Network errors: Confirm frontend VITE_API_BASE points to the backend URL reachable from the browser.

License
MIT (adjust as needed)
