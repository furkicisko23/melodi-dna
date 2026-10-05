from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import tempfile
import os
import numpy as np
import librosa
from difflib import SequenceMatcher

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

def clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))

def safe_load(path):
    y, sr = librosa.load(path, sr=22050, mono=True)
    if y.size == 0:
        raise ValueError("Ses dosyası boş.")
    y, _ = librosa.effects.trim(y, top_db=35)
    if y.size < sr * 0.25:
        raise ValueError("Ses çok kısa. En az yaklaşık 0,25 saniye gerekli.")
    # Normalize only for analysis; this does not alter the user's uploaded file.
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), sr

def pitch_track(y, sr):
    # Monophonic melody extraction. pyin returns NaN for uncertain frames.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=2048,
        hop_length=256,
        fill_na=np.nan
    )
    midi = librosa.hz_to_midi(f0)
    valid = np.isfinite(midi)
    if valid.sum() < 6:
        # Fallback: spectral centroid as a coarse pitch proxy.
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
        midi = librosa.hz_to_midi(np.maximum(centroid, 1.0))
        valid = np.isfinite(midi)
    idx = np.arange(len(midi))
    if valid.sum() >= 2:
        midi = np.interp(idx, idx[valid], midi[valid])
    else:
        midi = np.zeros_like(midi) + 60.0
    # Remove extreme frame-to-frame jumps caused by octave errors.
    for i in range(1, len(midi)):
        d = midi[i] - midi[i-1]
        if abs(d) > 12:
            midi[i] = midi[i-1] + np.sign(d) * min(abs(d), 7)
    return midi

def compress_sequence(seq, n=100):
    if len(seq) <= n:
        return seq
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, seq)

def interval_vector(midi):
    m = compress_sequence(midi, 100)
    return np.diff(m)

def contour_vector(midi):
    d = np.diff(compress_sequence(midi, 100))
    return np.sign(d)

def cosine_similarity(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    a -= np.mean(a); b -= np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return clamp((np.dot(a, b) / den + 1) * 50, 0, 100)

def interval_similarity(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    x = np.asarray(a[:n], dtype=float)
    y = np.asarray(b[:n], dtype=float)
    # Normalize local scale so transposition is naturally ignored.
    x = x - np.median(x)
    y = y - np.median(y)
    err = np.mean(np.minimum(np.abs(x-y), 12.0))
    return clamp(100 * np.exp(-err / 4.0))

def contour_similarity(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    x = np.asarray(a[:n])
    y = np.asarray(b[:n])
    # Sign agreement, with small motions treated as neutral.
    x = np.where(np.abs(x) < 0.12, 0, x)
    y = np.where(np.abs(y) < 0.12, 0, y)
    return float(np.mean(x == y) * 100)

def onset_intervals(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=256,
        backtrack=False, units="frames"
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=256)
    if len(times) < 3:
        # fallback to beat grid
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
        times = librosa.frames_to_time(beats, sr=sr, hop_length=256)
    if len(times) < 3:
        return np.array([])
    d = np.diff(times)
    d = d[(d > 0.04) & (d < 5.0)]
    if len(d) == 0:
        return np.array([])
    # Tempo invariance: compare ratios to median duration.
    return d / np.median(d)

def rhythm_similarity(y1, sr1, y2, sr2):
    a = onset_intervals(y1, sr1)
    b = onset_intervals(y2, sr2)
    if len(a) < 2 or len(b) < 2:
        return 50.0
    n = min(len(a), len(b), 80)
    a, b = a[:n], b[:n]
    err = np.mean(np.minimum(np.abs(a-b), 3.0))
    return clamp(100 * np.exp(-err / 1.15))

def quantized_intervals(midi):
    d = np.diff(compress_sequence(midi, 140))
    # Quantize semitone motion into a compact symbolic representation.
    q = np.round(np.clip(d, -12, 12)).astype(int)
    q = q[np.abs(q) >= 1]
    return q.tolist()

def ngrams(seq, k=4):
    if len(seq) < k:
        return set()
    return {tuple(seq[i:i+k]) for i in range(len(seq)-k+1)}

def motif_similarity(m1, m2):
    if len(m1) < 4 or len(m2) < 4:
        return 0.0
    scores = []
    for k in (3, 4, 5):
        A, B = ngrams(m1, k), ngrams(m2, k)
        if A and B:
            scores.append(100 * len(A & B) / max(1, len(A | B)))
    seq_score = SequenceMatcher(None, m1, m2, autojunk=False).ratio() * 100
    return clamp(0.65 * (max(scores) if scores else 0) + 0.35 * seq_score)

def analyze_pair(file1, file2):
    y1, sr1 = safe_load(file1)
    y2, sr2 = safe_load(file2)

    p1 = pitch_track(y1, sr1)
    p2 = pitch_track(y2, sr2)

    intervals = interval_similarity(interval_vector(p1), interval_vector(p2))
    contour = contour_similarity(contour_vector(p1), contour_vector(p2))
    rhythm = rhythm_similarity(y1, sr1, y2, sr2)
    motif = motif_similarity(quantized_intervals(p1), quantized_intervals(p2))

    overall = 0.35 * intervals + 0.25 * contour + 0.20 * rhythm + 0.20 * motif

    # Short preview data for the browser.
    def preview(p):
        p = compress_sequence(p, 80)
        base = float(np.median(p))
        return [round(float(x - base), 2) for x in p]

    return {
        "score": round(clamp(overall), 1),
        "components": {
            "interval": round(intervals, 1),
            "contour": round(contour, 1),
            "rhythm": round(rhythm, 1),
            "motif": round(motif, 1)
        },
        "melody1": preview(p1),
        "melody2": preview(p2),
        "weights": {"interval": 35, "contour": 25, "rhythm": 20, "motif": 20}
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/analyze")
def analyze():
    if "file1" not in request.files or "file2" not in request.files:
        return jsonify({"error": "İki ses dosyasını da seçmelisin."}), 400
    f1, f2 = request.files["file1"], request.files["file2"]
    if not f1.filename or not f2.filename:
        return jsonify({"error": "İki ses dosyasını da seçmelisin."}), 400

    ext1 = Path(f1.filename).suffix.lower()
    ext2 = Path(f2.filename).suffix.lower()
    if ext1 not in ALLOWED or ext2 not in ALLOWED:
        return jsonify({"error": "Desteklenen formatlar: WAV, MP3, FLAC, OGG, M4A, AAC."}), 400

    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "melody1" + ext1)
        p2 = os.path.join(td, "melody2" + ext2)
        f1.save(p1); f2.save(p2)
        try:
            result = analyze_pair(p1, p2)
        except Exception as e:
            return jsonify({
                "error": "Ses analiz edilemedi. Özellikle vokal/tek sesli ve temiz melodilerde daha iyi sonuç verir.",
                "detail": str(e)
            }), 422
    return jsonify(result)

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Dosya boyutu çok büyük. Her dosya en fazla 40 MB olabilir."}), 413

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
