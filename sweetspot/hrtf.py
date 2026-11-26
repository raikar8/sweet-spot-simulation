"""Head-related impulse responses (HRIRs).

Default: the MIT KEMAR dummy-head set (Gardner & Martin, MIT Media Lab, 1994),
710 directions at 44.1 kHz, shipped as assets/hrtf/mit_kemar.npz. The data may
be used freely with acknowledgement of its authors.

You can load your own HRTF from a SOFA file (SimpleFreeFieldHRIR), for example a
personalised set, with HRTF.from_sofa(path).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ASSETS = Path(__file__).resolve().parent.parent / "assets"
FS = 44100


def _unit(az_right_deg, el_deg):
    az = np.radians(az_right_deg)
    el = np.radians(el_deg)
    # x right, y front, z up
    return np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], axis=-1)


class HRTF:
    """A set of HRIRs indexed by direction. Azimuth convention: degrees, + = right."""

    def __init__(self, hrir: np.ndarray, az_right: np.ndarray, el: np.ndarray, fs: int, name: str):
        # hrir: (n_directions, taps, 2) with channel 0 = left ear
        if fs != FS:
            hrir = resample_poly(hrir, FS, int(fs), axis=1)
        self.hrir = hrir.astype(np.float32)
        self.az = np.asarray(az_right, float)
        self.el = np.asarray(el, float)
        self.dirs = _unit(self.az, self.el)
        self.fs = FS
        self.name = name
        self.taps = self.hrir.shape[1]

    @classmethod
    def kemar(cls) -> "HRTF":
        return _kemar()

    @classmethod
    def from_sofa(cls, path) -> "HRTF":
        import sofar  # optional dependency

        s = sofar.read_sofa(str(path), verify=False)
        ir = np.asarray(s.Data_IR)  # (M, R=2, N)
        pos = np.asarray(s.SourcePosition)  # az (CCW, deg), el (deg), r
        fs = int(np.asarray(s.Data_SamplingRate).ravel()[0])
        hrir = np.transpose(ir, (0, 2, 1))
        return cls(hrir, -pos[:, 0], pos[:, 1], fs, Path(path).stem)

    def nearest(self, az_right: float, el: float) -> np.ndarray:
        """HRIR pair (taps, 2) for the measured direction closest to (az, el)."""
        u = _unit(az_right, el)
        i = int(np.argmax(self.dirs @ u))
        return self.hrir[i]

    def stereo_pan(self, az_right: float, el: float) -> np.ndarray:
        """Plain stereo headphone panning (equal power by azimuth, no HRTF) as a 2-tap 'HRIR'."""
        lat = np.clip(np.sin(np.radians(az_right)) * np.cos(np.radians(el)), -1, 1)
        theta = (lat + 1) * np.pi / 4  # 0 = hard left, pi/2 = hard right
        h = np.zeros((self.taps, 2), np.float32)
        h[0, 0] = np.cos(theta)
        h[0, 1] = np.sin(theta)
        return h


@lru_cache(maxsize=1)
def _kemar() -> HRTF:
    d = np.load(ASSETS / "hrtf" / "mit_kemar.npz")
    # file stores SOFA-style azimuth (counter-clockwise, + = left)
    az_right = -d["az"]
    return HRTF(d["hrir"], az_right, d["el"], int(d["fs"]), "MIT KEMAR (generic dummy head)")
