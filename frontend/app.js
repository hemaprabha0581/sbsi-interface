const BACKEND_URL = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const recordBtn = document.getElementById("recordBtn");
const recordStatus = document.getElementById("recordStatus");
const playback = document.getElementById("playback");
const resultEl = document.getElementById("result");
const expectedLabel = document.getElementById("expectedLabel");
const accuracyEl = document.getElementById("accuracy");
const resetStatsBtn = document.getElementById("resetStatsBtn");

let mediaRecorder = null;
let recordedChunks = [];
let recording = false;

// Stats helpers
function getStats() {
  try {
    const s = JSON.parse(localStorage.getItem("sbsiStats") || "null");
    if (s && typeof s.total === "number" && typeof s.correct === "number") return s;
  } catch {}
  return { total: 0, correct: 0 };
}
function saveStats(s) { localStorage.setItem("sbsiStats", JSON.stringify(s)); }
function updateStats(correct) {
  const s = getStats();
  s.total += 1;
  if (correct) s.correct += 1;
  saveStats(s);
  updateAccuracyUI();
}
function updateAccuracyUI() {
  if (!accuracyEl) return;
  const s = getStats();
  if (s.total === 0) {
    accuracyEl.textContent = "Session accuracy: N/A";
  } else {
    const acc = ((s.correct / s.total) * 100).toFixed(1);
    accuracyEl.textContent = `Session accuracy: ${acc}% (${s.correct}/${s.total})`;
  }
}
resetStatsBtn && resetStatsBtn.addEventListener("click", () => {
  saveStats({ total: 0, correct: 0 });
  updateAccuracyUI();
  setResult("Stats reset.");
});
window.addEventListener("load", updateAccuracyUI);

function setResult(msg, ok = true) {
  resultEl.innerHTML = ok ? `<span class="ok">${msg}</span>` : `<span class="err">${msg}</span>`;
}

async function predictBlob(blob, filename = "audio.wav") {
  const formData = new FormData();
  formData.append("file", blob, filename);

  setResult("Sending to backend...");
  try {
    const resp = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    const conf = (data.confidence * 100).toFixed(1);
    const expected = expectedLabel && expectedLabel.value ? expectedLabel.value : "";
    let msg = `Predicted: ${data.predicted_class} (confidence: ${conf}%)`;
    if (expected) {
      if (data.predicted_class === expected) {
        msg += " — Correct!";
        updateStats(true);
      } else {
        msg += ` — Incorrect (expected: ${expected})`;
        updateStats(false);
      }
    } else {
      updateAccuracyUI();
    }
    setResult(msg, true);
  } catch (e) {
    console.error(e);
    setResult(`Error: ${e.message}`, false);
  }
}

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setResult("Please choose a WAV file first.", false);
    return;
  }
  if (!file.name.toLowerCase().endsWith(".wav")) {
    setResult("Please upload a .wav file.", false);
    return;
  }
  await predictBlob(file, file.name);
});

recordBtn.addEventListener("click", async () => {
  if (!recording) {
    // Start recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(recordedChunks, { type: "audio/webm" });
        playback.src = URL.createObjectURL(blob);
        recordStatus.textContent = "Converting to WAV...";
        try {
          const wavBlob = await blobToWav(blob);
          recordStatus.textContent = "Uploading...";
          await predictBlob(wavBlob, "recording.wav");
        } catch (e) {
          console.error(e);
          setResult("Could not convert/submit recording.", false);
        } finally {
          recordStatus.textContent = "Idle";
        }
      };
      mediaRecorder.start();
      recording = true;
      recordBtn.textContent = "Stop Recording";
      recordStatus.textContent = "Recording...";
    } catch (e) {
      console.error(e);
      setResult("Microphone permission denied or unavailable.", false);
    }
  } else {
    // Stop recording
    mediaRecorder && mediaRecorder.stop();
    recording = false;
    recordBtn.textContent = "Start Recording";
  }
});

async function blobToWav(blob) {
  const arrayBuf = await blob.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuf = await audioCtx.decodeAudioData(arrayBuf);

  // Convert to mono 16-bit PCM WAV
  const numChannels = audioBuf.numberOfChannels;
  const sampleRate = audioBuf.sampleRate;
  const length = audioBuf.length;

  // Merge down to mono
  let mono = new Float32Array(length);
  for (let ch = 0; ch < numChannels; ch++) {
    const data = audioBuf.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += data[i] / numChannels;
    }
  }

  const wavBuffer = encodeWAV(mono, sampleRate);
  return new Blob([wavBuffer], { type: "audio/wav" });
}

function encodeWAV(samples, sampleRate) {
  const bytesPerSample = 2; // 16-bit PCM
  const blockAlign = 1 * bytesPerSample; // mono
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  /* RIFF identifier */
  writeString(view, 0, "RIFF");
  /* RIFF chunk length */
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  /* RIFF type */
  writeString(view, 8, "WAVE");
  /* format chunk identifier */
  writeString(view, 12, "fmt ");
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * blockAlign, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, blockAlign, true);
  /* bits per sample */
  view.setUint16(34, 8 * bytesPerSample, true);
  /* data chunk identifier */
  writeString(view, 36, "data");
  /* data chunk length */
  view.setUint32(40, samples.length * bytesPerSample, true);

  // PCM conversion
  floatTo16BitPCM(view, 44, samples);

  return buffer;
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    output.setInt16(offset, s, true);
  }
}
