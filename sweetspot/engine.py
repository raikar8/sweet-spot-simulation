"""Render a scene (moving sound sources) through a speaker system to binaural audio at one seat."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .acoustics import BRIROptions, Calibration, calibrate, direct_info, speaker_brir
from .geometry import Listener, Room
from .hrtf import FS, HRTF
from .renderers import RenderContext, Renderer
from .scenes import Scene

BLOCK = 512  # samples between renderer updates for moving sources (11.6 ms)


@dataclass
class SystemSetup:
    room: Room
    layout: str
    cal_target: str | None = None   # a key of room.cal_points, "seat" (follow the listener) or None
    align: bool = True
    level: bool = True
    brir: BRIROptions = field(default_factory=BRIROptions)


class Seat:
    """A listener in a room with a speaker layout: holds the per-speaker BRIRs and calibration."""

    def __init__(self, setup: SystemSetup, listener: Listener, hrtf: HRTF):
        self.setup = setup
        self.room = setup.room
        self.listener = listener
        self.hrtf = hrtf
        self.speakers = setup.room.build_layout(setup.layout)
        self.ctx = RenderContext.build(self.room, self.speakers)
        self.brirs = [speaker_brir(self.room, s, listener, hrtf, setup.brir) for s in self.speakers]
        if setup.cal_target is None:
            cal_pt = self.room.ref3()
        elif setup.cal_target == "seat":
            cal_pt = listener.pos
        else:
            cal_pt = self.room.cal3(setup.cal_target)
        self.cal = calibrate(self.speakers, cal_pt, setup.align, setup.level)
        self.norm = _room_norm(self.room, self.speakers)

    # ------------------------------------------------------------ helpers
    def effective(self, renderer: Renderer):
        """Per-speaker constant (gain, delay) added by calibration for this renderer."""
        if renderer.uses_calibration:
            return self.cal.speaker_gain(self.speakers) * self.norm, self.cal.delays
        return np.full(len(self.speakers), self.norm), np.zeros(len(self.speakers))

    def system_ir(self, renderer: Renderer, source_pos: np.ndarray, include_sub: bool = False) -> np.ndarray:
        """Binaural impulse response (n, 2) for a static source rendered by `renderer` at this seat."""
        drv = renderer.drive(np.asarray(source_pos, float), self.ctx)
        cg, cd = self.effective(renderer)
        total_d = drv.delays + cd
        n = max(len(b) for b in self.brirs) + int(total_d.max() * FS) + 2
        out = np.zeros((n, 2), np.float64)
        for i, b in enumerate(self.brirs):
            g = drv.gains[i] * cg[i]
            if self.speakers[i].is_sub:
                if not include_sub:
                    continue
                g = 0.5 * cg[i]
            if g == 0:
                continue
            t = total_d[i] * FS
            i0 = int(t)
            fr = t - i0
            out[i0:i0 + len(b)] += (1 - fr) * g * b
            out[i0 + 1:i0 + 1 + len(b)] += fr * g * b
        return out

    def arrivals(self, renderer: Renderer, source_pos: np.ndarray):
        """Direct-sound arrival time (s) and amplitude per speaker at this seat."""
        drv = renderer.drive(np.asarray(source_pos, float), self.ctx)
        cg, cd = self.effective(renderer)
        rows = []
        for i, s in enumerate(self.speakers):
            r, g, off = direct_info(s, self.listener.pos)
            rows.append(dict(speaker=s.name, channel=s.channel, distance_m=r, added_delay_ms=1000 * (cd[i] + drv.delays[i]),
                             arrival_ms=1000 * (r / 343.0 + cd[i] + drv.delays[i]), amplitude=drv.gains[i] * cg[i] * g / r,
                             off_axis_deg=off, is_sub=s.is_sub))
        return rows

    # ------------------------------------------------------------ scene rendering
    def render(self, scene: Scene, renderer: Renderer, progress=None) -> np.ndarray:
        """Binaural (n, 2) render of the whole scene."""
        renderer.prepare(self.ctx)
        n_spk = len(self.speakers)
        L = scene.n_samples
        feeds = np.zeros((n_spk, L), np.float32)
        cg, cd = self.effective(renderer)
        sub_feed = np.zeros(L, np.float32)
        t_axis = np.arange(L)
        for k, tr in enumerate(scene.tracks):
            x = tr.signal.astype(np.float32) * tr.gain
            sub_feed += x * tr.lfe
            if tr.is_static:
                drv = renderer.drive(tr.position(0.0), self.ctx)
                for i in np.nonzero(drv.gains)[0]:
                    feeds[i] += _shift(x, drv.delays[i]) * drv.gains[i]
            else:
                nb = int(np.ceil(L / BLOCK)) + 1
                times = np.arange(nb) * BLOCK / FS
                G = np.zeros((nb, n_spk))
                D = np.zeros((nb, n_spk))
                for b, t in enumerate(times):
                    drv = renderer.drive(tr.position(t), self.ctx)
                    G[b] = drv.gains
                    D[b] = drv.delays
                for i in np.nonzero(G.any(axis=0))[0]:
                    g = np.interp(t_axis, times * FS, G[:, i]).astype(np.float32)
                    if D[:, i].any():
                        d = np.interp(t_axis, times * FS, D[:, i]) * FS
                        feeds[i] += g * np.interp(t_axis - d, t_axis, x, left=0, right=0).astype(np.float32)
                    else:
                        feeds[i] += g * x
            if progress:
                progress((k + 1) / (len(scene.tracks) + 1))
        out = np.zeros((L + max(len(b) for b in self.brirs) + int(cd.max() * FS) + 2, 2), np.float32)
        for i, b in enumerate(self.brirs):
            f = sub_feed * 0.5 if self.speakers[i].is_sub else feeds[i]
            if not f.any():
                continue
            f = _shift(f, cd[i]) * cg[i]
            y = fftconvolve(f[:, None], b, axes=0)
            out[: len(y)] += y[: len(out)]
        if progress:
            progress(1.0)
        return out[: L + int(0.3 * FS)]


def _shift(x: np.ndarray, delay_s: float) -> np.ndarray:
    k = int(round(delay_s * FS))
    if k <= 0:
        return x
    y = np.zeros_like(x)
    y[k:] = x[:-k]
    return y


def _room_norm(room: Room, speakers) -> float:
    """A fixed gain per room so that a calibrated channel has the same loudness at the reference seat."""
    cal = calibrate(speakers, room.ref3(), True, True)
    mains = [s for s in speakers if not s.is_sub]
    e = 0.0
    ch = mains[0].channel
    for s in mains:
        if s.channel == ch:
            r, g, _ = direct_info(s, room.ref3())
            e += (cal.channel_gain[ch] * g / r) ** 2
    return float(0.25 / np.sqrt(e))


def match_loudness(*signals: np.ndarray, peak: float = 0.89) -> list[np.ndarray]:
    """Scale several renders by one common factor so they keep their relative level and don't clip."""
    m = max(float(np.abs(s).max()) for s in signals) + 1e-12
    k = min(1.0, peak / m) if m > peak else peak / m * 0.7 if m < 0.05 else 1.0
    return [s * k for s in signals]
