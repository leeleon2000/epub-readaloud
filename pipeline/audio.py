from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read(path: Path) -> tuple[np.ndarray, int]:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a, sr


def duration(path: Path) -> float:
    i = sf.info(str(path))
    return i.frames / i.samplerate


def write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    tmp.replace(path)


def trim(audio: np.ndarray, sr: int, rel_db: float = -40.0, margin: float = 0.03) -> np.ndarray:
    """Cut leading/trailing silence relative to the take's peak, keeping a small margin."""
    if audio.size == 0:
        return audio
    mag = np.abs(audio)
    peak = float(mag.max())
    if peak <= 0.0:
        return audio[:0]
    idx = np.flatnonzero(mag > peak * 10 ** (rel_db / 20))
    a0 = max(0, int(idx[0] - margin * sr))
    a1 = min(audio.size, int(idx[-1] + margin * sr) + 1)
    return audio[a0:a1]


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0


def normalize(audio: np.ndarray, target_rms_db: float, peak_db: float) -> tuple[np.ndarray, float]:
    """Scale to the RMS target, but never past the peak ceiling. Returns (audio, gain_db)."""
    r, p = rms(audio), float(np.abs(audio).max()) if audio.size else 0.0
    if r <= 0 or p <= 0:
        return audio, 0.0
    gain = min(10 ** ((target_rms_db - db(r)) / 20), 10 ** ((peak_db - db(p)) / 20))
    return (audio * gain).astype(np.float32), db(gain)


def f0_track(audio: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0,
             frame: float = 0.04, hop: float = 0.01, threshold: float = 0.15) -> np.ndarray:
    """Pitch (Hz) of the voiced frames, YIN (cumulative mean normalized difference). Rough but
    enough to tell a typical male voice (85-155 Hz) from a female one (165-255 Hz)."""
    a = np.asarray(audio, np.float64)
    step = max(1, int(sr // 16000))
    if step > 1:                                   # crude low-pass + decimate to ~16 kHz
        a = np.convolve(a, np.ones(step) / step, mode="same")[::step]
        sr = sr / step
    w, h = int(frame * sr), max(1, int(hop * sr))
    tmin, tmax = max(2, int(sr / fmax)), int(sr / fmin)
    n = w + tmax
    if a.size < n:
        return np.zeros(0)
    starts = range(0, a.size - n, h)
    frames = np.stack([a[i:i + n] for i in starts])
    e = np.sqrt(np.mean(frames[:, :w] ** 2, axis=1))
    frames = frames[e > e.max() * 10 ** (-30 / 20)]          # skip silence and breaths
    if not len(frames):
        return np.zeros(0)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(frames, size) * np.conj(np.fft.rfft(frames[:, :w], size)), size)[:, :tmax + 1]
    sq = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    e_shift = sq[:, w:w + tmax + 1] - sq[:, :tmax + 1]        # energy of the window shifted by tau
    d = sq[:, w:w + 1] + e_shift - 2 * corr
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, tmax + 1) / np.maximum(cum, 1e-12)
    out = []
    for row in cmnd:
        below = np.flatnonzero(row[tmin:] < threshold)
        if not below.size:
            continue
        t = tmin + below[0]
        while t + 1 <= tmax and row[t + 1] < row[t]:        # walk down to the local minimum
            t += 1
        if 0 < t < tmax:                                    # parabolic interpolation
            y0, y1, y2 = row[t - 1], row[t], row[t + 1]
            den = y0 - 2 * y1 + y2
            t = t + (0.5 * (y0 - y2) / den if den else 0.0)
        out.append(sr / t)
    return np.asarray(out)


def median_f0(audio: np.ndarray, sr: int) -> float:
    f = f0_track(audio, sr)
    return float(np.median(f)) if f.size >= 10 else 0.0
