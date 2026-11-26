"""Objective metrics computed from what actually reaches the two ears.

All metrics take the binaural impulse response h (n, 2) of the whole system
(renderer + calibration + room + head) for one source position at one seat, and
compare it with the ideal: a single real source at that position.

Localisation (lateral error)
    The precedence effect makes us localise on the first ~1 ms of sound. We take
    a window from the first arrival to 1.5 ms after it, measure the interaural
    time difference (ITD, from each ear's onset) and level difference
    (ILD, 1.5 to 6 kHz), and find the horizontal direction whose HRTF has the most
    similar ITD and ILD. The lateral angle of that direction (left-right only, so
    front/back confusions don't count) is compared with the lateral angle of the
    intended source as seen from this seat.

Arrival spread       last minus first direct-sound arrival among speakers within 20 dB of the loudest.
Level                energy in the first 50 ms, dB (maps compare it with the reference seat).
Colouration          standard deviation (dB) of the difference between the system's and the ideal
                     magnitude spectrum in 1/3-octave bands 200 Hz to 8 kHz, first 25 ms. Comb
                     filtering from several speakers playing the same sound raises it.
Envelopment          1 - IACC of a diffuse bed (eight uncorrelated sources around the room), 0 to 80 ms.
                     Higher is more enveloping.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..hrtf import FS, HRTF

_LP = butter(4, 1500, "lowpass", fs=FS, output="sos")
_BP = butter(4, [1500, 6000], "bandpass", fs=FS, output="sos")
MAXLAG = int(0.001 * FS)


def onset(h: np.ndarray, rel: float = 0.1) -> int:
    e = np.abs(h).max(axis=1)
    idx = np.nonzero(e >= rel * e.max())[0]
    return int(idx[0]) if len(idx) else 0


def _itd_ild(seg: np.ndarray) -> tuple[float, float]:
    """ITD (us, > 0 = right ear leads, i.e. source on the right) and ILD (dB, > 0 = right louder).

    ITD is measured from each ear's onset (first sample above 20% of that ear's peak),
    which follows the first wavefront even when a second speaker arrives a fraction
    of a millisecond later; that is how the precedence effect behaves.
    """
    pad = np.zeros((256, 2))
    x = np.vstack([pad, seg, pad])
    env = np.abs(x)
    on = [int(np.argmax(env[:, k] >= 0.2 * env[:, k].max())) for k in (0, 1)]
    # refine to sub-sample with the cross-correlation peak near the onset difference
    itd = (on[0] - on[1]) / FS * 1e6
    hi = sosfiltfilt(_BP, x, axis=0)
    ild = 10 * np.log10((np.sum(hi[:, 1] ** 2) + 1e-12) / (np.sum(hi[:, 0] ** 2) + 1e-12))
    return float(np.clip(itd, -900, 900)), float(ild)


def precedence_window(h: np.ndarray, before_ms=0.3, after_ms=1.5) -> np.ndarray:
    t0 = onset(h)
    a = max(0, t0 - int(before_ms * FS / 1000))
    return h[a: t0 + int(after_ms * FS / 1000)]


@lru_cache(maxsize=4)
def _cue_table(hrtf_id: int):
    hrtf = _HRTFS[hrtf_id]
    az = np.arange(-90, 91, 2.5)
    rows = []
    for a in az:
        h = hrtf.nearest(a, 0.0).astype(float)
        itd, ild = _itd_ild(precedence_window(h))
        rows.append((itd, ild))
    return az, np.array(rows)


_HRTFS: dict = {}


def lateral_angle(h: np.ndarray, hrtf: HRTF) -> float:
    """Estimated perceived lateral angle (deg, -90 left .. +90 right) from the first wavefront."""
    _HRTFS[id(hrtf)] = hrtf
    az, table = _cue_table(id(hrtf))
    itd, ild = _itd_ild(precedence_window(h))
    cost = ((itd - table[:, 0]) / 80.0) ** 2 + ((ild - table[:, 1]) / 3.0) ** 2
    return float(az[int(np.argmin(cost))])


def target_lateral(az_deg: float, el_deg: float) -> float:
    return float(np.degrees(np.arcsin(np.sin(np.radians(az_deg)) * np.cos(np.radians(el_deg)))))


def level_db(h: np.ndarray, ms: float = 50) -> float:
    t0 = onset(h)
    seg = h[t0: t0 + int(ms * FS / 1000)]
    return float(10 * np.log10(np.sum(seg ** 2) + 1e-12))


_EDGES = 200 * 2 ** (np.arange(0, 16) / 3)  # 200 Hz .. ~6.3k, 1/3 octave


def _bands(seg: np.ndarray) -> np.ndarray:
    n = 4096
    X = np.abs(np.fft.rfft(seg, n=n, axis=0)) ** 2
    f = np.fft.rfftfreq(n, 1 / FS)
    out = []
    for lo in _EDGES:
        hi = lo * 2 ** (1 / 3)
        m = (f >= lo) & (f < hi)
        out.append(X[m].mean() if m.any() else 1e-12)
    return 10 * np.log10(np.array(out) + 1e-12)


def colouration_db(h: np.ndarray, ideal: np.ndarray, ms: float = 25) -> float:
    t0 = onset(h)
    a = h[t0: t0 + int(ms * FS / 1000)]
    b = ideal[onset(ideal): onset(ideal) + int(ms * FS / 1000)]
    d = (_bands(a[:, 0]) + _bands(a[:, 1])) / 2 - (_bands(b[:, 0]) + _bands(b[:, 1])) / 2
    return float(np.std(d - d.mean()))


def envelopment(irs: list[np.ndarray], ms: float = 80) -> float:
    """1 - IACC for a diffuse bed made of uncorrelated sources with these system IRs."""
    t0 = min(onset(h) for h in irs)
    n = int(ms * FS / 1000)
    cc = np.zeros(2 * MAXLAG + 1)
    el = er = 0.0
    for h in irs:
        s = h[t0: t0 + n]
        l, r = s[:, 0], s[:, 1]
        full = np.correlate(l, r, "full")
        mid = len(r) - 1
        cc += full[mid - MAXLAG: mid + MAXLAG + 1]
        el += np.sum(l ** 2)
        er += np.sum(r ** 2)
    return float(1 - np.abs(cc).max() / np.sqrt(el * er + 1e-12))
