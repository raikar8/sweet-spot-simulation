"""Sound propagation from each loudspeaker to the two ears.

For a fixed listener, every loudspeaker has a binaural room impulse response
(BRIR): what the left and right ear receive when that speaker emits a click.
It is built from:

1. the direct path and image-source reflections off the six walls of the
   shoebox (Allen & Berkley, 1979), each with delay r/c, 1/r spreading, the
   wall reflection coefficients, air absorption and the speaker's directivity
   towards that path;
2. the HRIR for the direction the path arrives from (relative to the head);
3. a statistical late-reverberation tail (exponentially decaying, decorrelated
   noise with the room's RT60);
4. the speaker's band filter (tweeter / woofer crossover, subwoofer low-pass).

Once the BRIRs for a seat exist, any rendering algorithm only has to decide
what each speaker plays: the ear signals are sum_i (feed_i * BRIR_i).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from .geometry import SPEED_OF_SOUND as C, Listener, Room, Speaker
from .hrtf import FS, HRTF

CROSSOVER_HZ = 2500.0
SUB_HZ = 100.0


@dataclass
class Path:
    pos: np.ndarray      # (image) source position
    axis: np.ndarray     # (image) speaker axis
    gain: float          # product of wall reflection coefficients
    order: int


def image_sources(room: Room, spk: Speaker, max_order: int) -> list[Path]:
    """Image sources of a speaker in the shoebox, up to max_order reflections (Allen & Berkley).

    Per dimension with walls at lo and hi (length L), an image is (1 - 2q) * x + 2 n L
    (x measured from lo) for q in {0, 1} and integer n. It reflects |n - q| times off
    the lo wall and |n| times off the hi wall, and q = 1 mirrors the speaker's axis.
    """
    w = room.walls
    bounds = [(room.x0, room.x1, w.x0, w.x1), (room.y0, room.y1, w.y0, w.y1), (0.0, room.height, w.z0, w.z1)]
    per_dim = []
    for d, (lo, hi, r_lo, r_hi) in enumerate(bounds):
        L = hi - lo
        x = spk.pos[d] - lo
        opts = []
        for q in (0, 1):
            for n in range(-max_order, max_order + 1):
                n_lo, n_hi = abs(n - q), abs(n)
                if n_lo + n_hi > max_order:
                    continue
                opts.append(((1 - 2 * q) * x + 2 * n * L + lo, q, n_lo + n_hi, (r_lo ** n_lo) * (r_hi ** n_hi)))
        per_dim.append(opts)
    paths = []
    for (px, qx, ox, gx), (py, qy, oy, gy), (pz, qz, oz, gz) in product(*per_dim):
        order = ox + oy + oz
        if order > max_order:
            continue
        axis = spk.axis * np.array([1 - 2 * qx, 1 - 2 * qy, 1 - 2 * qz])
        paths.append(Path(np.array([px, py, pz]), axis, gx * gy * gz, order))
    return paths


def directivity(spk: Speaker, cos_off: float) -> tuple[float, float]:
    """(broadband gain, low-pass cutoff Hz) of the speaker in a direction cos_off off its axis."""
    c = (1.0 + cos_off) / 2.0
    if spk.band in ("full", "high"):
        return 0.15 + 0.85 * c ** 0.8, float(np.clip(20000 * c ** 1.5, 2500, 20000))
    if spk.band == "low":
        return 0.5 + 0.5 * c ** 0.3, 20000.0
    return 1.0, 20000.0


def _one_pole(x: np.ndarray, fc: float) -> np.ndarray:
    if fc >= 19000:
        return x
    a = np.exp(-2 * np.pi * fc / FS)
    return lfilter([1 - a], [1, -a], x, axis=0)


@lru_cache(maxsize=8)
def _band_sos(band: str):
    if band == "high":
        return butter(4, CROSSOVER_HZ, "highpass", fs=FS, output="sos")
    if band == "low":
        return butter(4, CROSSOVER_HZ, "lowpass", fs=FS, output="sos")
    if band == "sub":
        return butter(4, SUB_HZ, "lowpass", fs=FS, output="sos")
    return None


def direct_info(spk: Speaker, point: np.ndarray) -> tuple[float, float, float]:
    """(distance m, broadband directivity gain, off-axis angle deg) for the direct path to point."""
    d = point - spk.pos
    r = max(0.15, float(np.linalg.norm(d)))
    cos_off = float(np.dot(spk.axis, d / r))
    g, _ = directivity(spk, cos_off)
    return r, g, float(np.degrees(np.arccos(np.clip(cos_off, -1, 1))))


@dataclass
class BRIROptions:
    reflections: bool = True
    late_tail: bool = True
    head_model: str = "hrtf"  # "hrtf" or "stereo" (plain stereo panning, no HRTF)
    max_order: int | None = None


def speaker_brir(room: Room, spk: Speaker, lis: Listener, hrtf: HRTF, opt: BRIROptions,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Binaural impulse response (n, 2) from one speaker to the listener's ears."""
    order = (room.reflection_order if opt.max_order is None else opt.max_order) if opt.reflections else 0
    paths = image_sources(room, spk, order)
    tail_len = int(FS * room.rt60 * 1.2) if opt.late_tail else 0
    max_r = max(np.linalg.norm(p.pos - lis.pos) for p in paths)
    n = int(max_r / C * FS) + hrtf.taps + tail_len + 64
    out = np.zeros((n, 2), np.float64)
    for p in paths:
        d = lis.pos - p.pos
        r = max(0.15, float(np.linalg.norm(d)))
        cos_off = float(np.dot(p.axis, d / r))
        g_dir, fc = directivity(spk, cos_off)
        # air absorption + a little high-frequency loss per wall bounce
        fc = min(fc, 20000.0 / (1 + 0.004 * r), 14000.0 / (1 + 0.35 * p.order) if p.order else 20000.0)
        amp = p.gain * g_dir / r
        az, el, _ = lis.to_head(p.pos)
        h = hrtf.nearest(az, el) if opt.head_model == "hrtf" else hrtf.stereo_pan(az, el)
        h = _one_pole(h.astype(np.float64), fc) * amp
        t = r / C * FS
        i0 = int(t)
        frac = t - i0
        m = h.shape[0]
        out[i0:i0 + m] += (1 - frac) * h
        out[i0 + 1:i0 + 1 + m] += frac * h
    if tail_len:
        rng = rng or np.random.default_rng(abs(hash(spk.name)) % 2**32)
        r_direct = max(0.15, float(np.linalg.norm(lis.pos - spk.pos)))
        t_mix = max(0.004, 1.5 * room_mean_free_path(room) / C * (order + 1))
        start = int((r_direct / C + t_mix) * FS)
        k = np.arange(tail_len)
        env = np.exp(-6.9 * k / (room.rt60 * FS))
        noise = rng.standard_normal((tail_len, 2)) * env[:, None]
        noise = _one_pole(noise, 7000.0)
        # diffuse level: energy ~ 1/rc^2 where rc is the critical distance
        rc = critical_distance(room)
        noise *= (1.0 / rc) / np.sqrt(np.sum(noise ** 2) / 2 + 1e-12) * 0.5
        if opt.head_model != "hrtf":
            noise[:, 1] = noise[:, 0] * 0.3 + noise[:, 1] * 0.7
        end = min(n, start + tail_len)
        out[start:end] += noise[: end - start]
    sos = _band_sos(spk.band)
    if sos is not None:
        out = sosfilt(sos, out, axis=0)
    return out.astype(np.float32)


def room_mean_free_path(room: Room) -> float:
    V = (room.x1 - room.x0) * (room.y1 - room.y0) * room.height
    S = 2 * ((room.x1 - room.x0) * (room.y1 - room.y0) + (room.x1 - room.x0) * room.height
             + (room.y1 - room.y0) * room.height)
    return 4 * V / S


def critical_distance(room: Room) -> float:
    """Distance at which direct and reverberant energy are equal (Sabine, Q = 2)."""
    V = (room.x1 - room.x0) * (room.y1 - room.y0) * room.height
    return float(0.057 * np.sqrt(2 * V / room.rt60))


# ---------------------------------------------------------------- calibration
@dataclass
class Calibration:
    delays: np.ndarray        # seconds, per speaker
    channel_gain: dict        # channel -> linear gain
    point: np.ndarray

    def speaker_gain(self, speakers) -> np.ndarray:
        return np.array([self.channel_gain.get(s.channel, 1.0) for s in speakers])


def calibrate(speakers: list[Speaker], point: np.ndarray, align: bool = True, level: bool = True) -> Calibration:
    """Time alignment and per-channel level trim at one point (the way installers tune a room).

    Delay: nearer speakers are delayed so every direct sound arrives together at `point`.
    Level: each channel (a whole surround array counts as one) gets the same energy at `point`.
    """
    info = [direct_info(s, point) for s in speakers]
    r = np.array([i[0] for i in info])
    delays = (r.max() - r) / C if align else np.zeros(len(speakers))
    energy: dict = {}
    for s, (ri, gi, _) in zip(speakers, info):
        energy[s.channel] = energy.get(s.channel, 0.0) + (gi / ri) ** 2
    mains = [c for c in energy if c != "LFE"]
    target = float(np.median([energy[c] for c in mains])) if mains else 1.0
    gains = {}
    for c, e in energy.items():
        gains[c] = float(np.sqrt((0.5 * target if c == "LFE" else target) / e)) if level else 1.0
    return Calibration(delays, gains, point)
