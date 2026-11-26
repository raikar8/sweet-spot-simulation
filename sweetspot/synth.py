"""Procedurally generated sounds, so the lab ships with no third-party audio.

Every function returns a mono float32 array at 44.1 kHz. Voices are pre-rendered
with the espeak-ng speech synthesiser into assets/sounds/ by tools/make_assets.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

from .hrtf import FS

SOUNDS = Path(__file__).resolve().parent.parent / "assets" / "sounds"


def _rng(seed):
    return np.random.default_rng(seed)


def _bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=FS, output="sos"), x)


def _lp(x, f, order=2):
    return sosfilt(butter(order, f, "lowpass", fs=FS, output="sos"), x)


def _hp(x, f, order=2):
    return sosfilt(butter(order, f, "highpass", fs=FS, output="sos"), x)


def _norm(x, peak=0.5):
    return (x / (np.abs(x).max() + 1e-9) * peak).astype(np.float32)


def silence(sec):
    return np.zeros(int(sec * FS), np.float32)


def pink(n, seed=0):
    w = _rng(seed).standard_normal(n)
    f = np.fft.rfft(w)
    k = np.arange(len(f))
    k[0] = 1
    return _norm(np.fft.irfft(f / np.sqrt(k), n))


def pink_bursts(sec, on=0.25, off=0.25, seed=0):
    x = pink(int(sec * FS), seed)
    t = np.arange(len(x)) / FS
    env = ((t % (on + off)) < on).astype(float)
    env = np.convolve(env, np.ones(220) / 220, "same")
    return _norm(x * env)


# ---------------------------------------------------------------- voices
@lru_cache(maxsize=32)
def voice(name: str) -> np.ndarray:
    """A pre-rendered speech line from assets/sounds/<name>.wav (falls back to vowel babble)."""
    f = SOUNDS / f"{name}.wav"
    if f.exists():
        fs, x = wavfile.read(f)
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x /= np.abs(x).max() + 1e-9
        if fs != FS:
            x = resample_poly(x, FS, fs)
        return _norm(x, 0.6)
    return babble(2.5, seed=abs(hash(name)) % 1000)


def babble(sec, f0=120, seed=0):
    """Synthetic vowel sequence (used only if the speech assets are missing)."""
    rng = _rng(seed)
    n = int(sec * FS)
    t = np.arange(n) / FS
    f = f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))
    ph = 2 * np.pi * np.cumsum(f) / FS
    x = sum(np.sin(k * ph) / k for k in range(1, 30))
    formants = rng.choice([500, 700, 300, 400], size=int(sec * 4) + 1)
    out = np.zeros(n)
    for i, F in enumerate(formants):
        a, b = int(i * FS / 4), min(n, int((i + 1) * FS / 4))
        out[a:b] = _bp(x[a:b], F * 0.7, F * 1.6) + 0.5 * _bp(x[a:b], 1800, 2800)
    return _norm(out * (np.sin(np.pi * np.clip(t * 4 % 1, 0, 1)) ** 0.5))


# ---------------------------------------------------------------- nature
def rain(sec, seed=1, intensity=1.0):
    rng = _rng(seed)
    n = int(sec * FS)
    bed = _hp(rng.standard_normal(n), 1500) * 0.15
    drops = np.zeros(n)
    for i in rng.integers(0, n - 400, int(sec * 300 * intensity)):
        L = rng.integers(40, 160)
        drops[i:i + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * rng.uniform(0.2, 1)
    return _norm(bed + _hp(drops, 800), 0.4)


def wind(sec, seed=2):
    rng = _rng(seed)
    n = int(sec * FS)
    x = _lp(rng.standard_normal(n), 600)
    t = np.arange(n) / FS
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t + 1) * np.sin(2 * np.pi * 0.037 * t)
    return _norm(x * (0.3 + gust), 0.4)


def stream(sec, seed=3):
    rng = _rng(seed)
    n = int(sec * FS)
    x = _bp(rng.standard_normal(n), 300, 3000)
    t = np.arange(n) / FS
    mod = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * (3 + rng.uniform(-1, 1)) * t)) * rng.uniform(0.6, 1, n)
    return _norm(x * mod, 0.4)


def bird(sec, seed=4, base=3500):
    """A songbird: bursts of fast frequency-modulated chirps."""
    rng = _rng(seed)
    n = int(sec * FS)
    out = np.zeros(n)
    t0 = rng.uniform(0.1, 1.0)
    while t0 < sec - 1:
        for _ in range(rng.integers(3, 8)):
            L = int(rng.uniform(0.04, 0.12) * FS)
            tt = np.arange(L) / FS
            f = base * rng.uniform(0.8, 1.3) + rng.uniform(-1500, 1500) * tt / tt[-1] + 400 * np.sin(2 * np.pi * rng.uniform(20, 60) * tt)
            ch = np.sin(2 * np.pi * np.cumsum(f) / FS) * np.sin(np.pi * tt / tt[-1]) ** 2
            i = int(t0 * FS)
            if i + L < n:
                out[i:i + L] += ch
            t0 += L / FS + rng.uniform(0.02, 0.08)
        t0 += rng.uniform(0.8, 2.5)
    return _norm(out, 0.35)


# ---------------------------------------------------------------- cinema effects
def thunder(sec, at=1.0, seed=5):
    rng = _rng(seed)
    n = int(sec * FS)
    out = np.zeros(n)
    i = int(at * FS)
    L = min(n - i, int(4 * FS))
    t = np.arange(L) / FS
    crack = _hp(rng.standard_normal(L), 1500) * np.exp(-t / 0.08)
    rumble = _lp(rng.standard_normal(L), 180) * (np.exp(-t / 1.4) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t) ** 2))
    out[i:i + L] = 0.4 * crack + 3.0 * rumble
    return _norm(out, 0.8)


def explosion(sec, at=1.0, seed=6):
    rng = _rng(seed)
    n = int(sec * FS)
    out = np.zeros(n)
    i = int(at * FS)
    L = min(n - i, int(3 * FS))
    t = np.arange(L) / FS
    body = _lp(rng.standard_normal(L), 2500) * np.exp(-t / 0.35)
    boom = np.sin(2 * np.pi * (55 - 25 * t) * t) * np.exp(-t / 0.6)
    debris = np.zeros(L)
    for j in rng.integers(int(0.2 * FS), L - 500, 80):
        debris[j:j + 300] += _hp(rng.standard_normal(300), 2000) * np.exp(-np.arange(300) / 60) * 0.3
    out[i:i + L] = body + 1.5 * boom + debris
    return _norm(out, 0.9)


def jet(sec, seed=7):
    """A jet engine: broadband roar plus a turbine whine."""
    rng = _rng(seed)
    n = int(sec * FS)
    t = np.arange(n) / FS
    roar = _lp(rng.standard_normal(n), 1800) + 0.3 * _hp(rng.standard_normal(n), 3000)
    whine = 0.15 * np.sin(2 * np.pi * 2900 * t + 3 * np.sin(2 * np.pi * 0.3 * t))
    return _norm(roar + whine, 0.6)


def engine_passby(sec, seed=8):
    """A truck: low engine harmonics plus tyre noise."""
    rng = _rng(seed)
    n = int(sec * FS)
    t = np.arange(n) / FS
    f0 = 38 + 4 * np.sin(2 * np.pi * 0.2 * t)
    ph = 2 * np.pi * np.cumsum(f0) / FS
    eng = sum(np.sin(k * ph) / k ** 0.7 for k in range(1, 18))
    tyre = _bp(rng.standard_normal(n), 400, 2500) * 0.6
    return _norm(eng * 0.4 + tyre, 0.6)


def footsteps(sec, rate=1.8, start=0.5, seed=9):
    rng = _rng(seed)
    n = int(sec * FS)
    out = np.zeros(n)
    t = start
    while t < sec - 0.3:
        i = int(t * FS)
        L = 2500
        k = np.arange(L)
        thud = _lp(rng.standard_normal(L), 600) * np.exp(-k / 300) + 0.3 * _hp(rng.standard_normal(L), 3000) * np.exp(-k / 80)
        out[i:i + L] += thud
        t += 1 / rate * rng.uniform(0.92, 1.08)
    return _norm(out, 0.5)


def pad(sec, notes=(48, 55, 60, 64), seed=10):
    """A soft synth chord, like a film score bed."""
    n = int(sec * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for m in notes:
        f = 440 * 2 ** ((m - 69) / 12)
        for det in (-0.3, 0.3):
            x += np.sin(2 * np.pi * (f + det) * t) + 0.3 * np.sin(4 * np.pi * (f + det) * t)
    env = np.minimum(1, t / 2) * np.minimum(1, (sec - t) / 2)
    return _norm(_lp(x, 2500) * env, 0.3)


def road_noise(sec, seed=11):
    rng = _rng(seed)
    n = int(sec * FS)
    return _norm(_lp(rng.standard_normal(n), 250) + 0.15 * _bp(rng.standard_normal(n), 800, 2000), 0.4)


def chime(sec, at=0.5):
    n = int(sec * FS)
    out = np.zeros(n)
    for k, (f, d) in enumerate([(880, 0), (1318.5, 0.18)]):
        i = int((at + d) * FS)
        L = min(n - i, int(0.8 * FS))
        tt = np.arange(L) / FS
        out[i:i + L] += np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.25)
    return _norm(out, 0.5)
