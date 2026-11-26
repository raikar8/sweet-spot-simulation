"""Content scenes: dialogue, nature ambience and cinema effects, placed and moving in the room.

A Scene is a list of Tracks. Each track is a mono signal plus a position over
time (keyframes). Renderers turn the position into speaker gains; the engine
does the rest. Add your own scenes with Scene / Track, or load your own WAVs
with Track.from_wav.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import synth as sy
from .hrtf import FS


@dataclass
class Track:
    name: str
    signal: np.ndarray
    keys: list                     # [(t_seconds, (x, y, z)), ...]; one key = static
    gain: float = 1.0
    lfe: float = 0.0               # extra send to the subwoofer (explosions, thunder)
    kind: str = "effect"           # "dialogue", "ambience", "effect", "music"

    @property
    def is_static(self) -> bool:
        return len(self.keys) == 1

    def position(self, t: float) -> np.ndarray:
        if self.is_static:
            return np.asarray(self.keys[0][1], float)
        ts = np.array([k[0] for k in self.keys])
        ps = np.array([k[1] for k in self.keys], float)
        return np.array([np.interp(t, ts, ps[:, d]) for d in range(3)])

    @classmethod
    def from_wav(cls, path, keys, **kw) -> "Track":
        from scipy.io import wavfile
        from scipy.signal import resample_poly

        fs, x = wavfile.read(path)
        x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x /= np.abs(x).max() + 1e-9
        if fs != FS:
            x = resample_poly(x, FS, fs).astype(np.float32)
        return cls(kw.pop("name", str(path)), x * 0.6, keys, **kw)


@dataclass
class Scene:
    name: str
    description: str
    seconds: float
    tracks: list = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.seconds * FS)

    def add(self, track: Track, at: float = 0.0) -> Track:
        """Place a track's signal starting at `at` seconds on the scene timeline."""
        buf = np.zeros(self.n_samples, np.float32)
        i = int(at * FS)
        x = track.signal[: max(0, self.n_samples - i)]
        buf[i:i + len(x)] = x
        track.signal = buf
        self.tracks.append(track)
        return track

    def ambience(self, name, make: Callable[[int], np.ndarray], center, radius, n=6, height=0.0, gain=0.3, kind="ambience"):
        """A diffuse bed: n decorrelated copies spread on a circle around `center`."""
        for k in range(n):
            a = 2 * np.pi * (k + 0.5) / n
            p = (center[0] + radius * np.sin(a), center[1] - radius * np.cos(a), center[2] + height)
            self.add(Track(f"{name} {k+1}", make(k), [(0, p)], gain=gain / np.sqrt(n), kind=kind))


# ---------------------------------------------------------------- cinema scenes
SCREEN_Z = 4.0


def cinema_dialogue() -> Scene:
    sc = Scene("Two characters talking", "Maya on screen left, Leo on screen right walks to the centre while he talks. "
               "Dialogue is the hardest test: your eyes see the actor, so the voice must come from the same spot.", 16)
    sc.add(Track("Maya (screen left)", sy.voice("maya_1"), [(0, (-4.5, 0.8, SCREEN_Z))], kind="dialogue"), at=0.3)
    sc.add(Track("Leo (screen right)", sy.voice("leo_1"), [(0, (4.5, 0.8, SCREEN_Z))], kind="dialogue"), at=4.3)
    sc.add(Track("Maya (screen left)", sy.voice("maya_2"), [(0, (-4.5, 0.8, SCREEN_Z))], kind="dialogue"), at=8.2)
    sc.add(Track("Leo walking to centre", sy.voice("leo_2"),
                 [(0, (5.5, 0.8, SCREEN_Z)), (10.5, (5.5, 0.8, SCREEN_Z)), (14.5, (0.0, 0.8, SCREEN_Z))], kind="dialogue"), at=10.4)
    sc.add(Track("Leo's footsteps", sy.footsteps(16, start=10.3),
                 [(0, (5.5, 0.8, 2.0)), (10.3, (5.5, 0.8, 2.0)), (14.5, (0.0, 0.8, 2.0))], gain=0.5), at=0)
    sc.ambience("Night wind", lambda k: sy.wind(16, seed=20 + k), (0, 14, 4), 12, n=6, gain=0.25)
    sc.add(Track("Score (low pad)", sy.pad(16), [(0, (0, 0.8, SCREEN_Z))], gain=0.35, kind="music"))
    return sc


def cinema_forest() -> Scene:
    sc = Scene("Forest at dawn", "Birds all around you, a stream front right, wind in the trees and one bird flying "
               "over your head from the screen to the back wall. Tests envelopment and the surround arrays.", 18)
    sc.ambience("Wind in the trees", lambda k: sy.wind(18, seed=30 + k), (0, 14, 4), 12, n=8, gain=0.35)
    spots = [(-7, 3, 6), (6, 2, 7), (-8.5, 15, 6), (8.5, 20, 6), (-4, 27, 6), (3, 27, 7)]
    for k, p in enumerate(spots):
        sc.add(Track(f"Bird {k+1}", sy.bird(18, seed=40 + k, base=2800 + 300 * k), [(0, p)], gain=0.55))
    sc.add(Track("Stream (front right)", sy.stream(18), [(0, (6.5, 1.0, 1.5))], gain=0.5))
    sc.add(Track("Bird flying overhead", sy.bird(6, seed=99, base=4200),
                 [(0, (0, 0.8, 6)), (6, (0, 27.5, 7))], gain=0.8), at=9)
    return sc


def cinema_storm() -> Scene:
    sc = Scene("Storm and flyover", "Rain everywhere, thunder rolling from left to right behind you, a jet flying "
               "from the screen over your head to the back, then an explosion on screen with deep bass.", 18)
    sc.ambience("Rain", lambda k: sy.rain(18, seed=50 + k), (0, 14, 4), 12, n=8, gain=0.4)
    sc.add(Track("Thunder roll", sy.thunder(18, at=1.0),
                 [(0, (-9, 24, 6)), (1.0, (-9, 24, 6)), (4.5, (9, 24, 6))], gain=0.8, lfe=0.6))
    sc.add(Track("Jet flyover", sy.jet(5),
                 [(0, (0, 0.5, 5)), (2.5, (0, 14, 9)), (5, (0, 28, 6))], gain=0.9, lfe=0.3), at=6)
    sc.add(Track("Explosion on screen", sy.explosion(18, at=13.0), [(0, (-1.5, 0.6, 3))], gain=1.0, lfe=1.0))
    return sc


def cinema_test() -> Scene:
    sc = Scene("Calibration test", "Pink-noise bursts walking around the room, then a centred voice. "
               "Good for judging where the image lands.", 14)
    path = [(0, (-6, 0.6, 4)), (2, (0, 0.6, 4)), (4, (6, 0.6, 4)), (6.5, (8.8, 17, 5)), (8.5, (0, 27.8, 5)), (10, (-8.8, 17, 5))]
    sc.add(Track("Noise walking around", sy.pink_bursts(10), path, gain=0.7))
    sc.add(Track("Centre voice", sy.voice("test_voice"), [(0, (0, 0.6, 4))], kind="dialogue"), at=10.2)
    return sc


# ---------------------------------------------------------------- car scenes
def car_conversation() -> Scene:
    sc = Scene("Radio, navigation and a phone call", "A radio presenter in the centre of the dash, a navigation "
               "prompt from the left, and a phone call. Road noise underneath.", 16)
    sc.ambience("Road noise", lambda k: sy.road_noise(16, seed=60 + k), (0, 1.4, 0.4), 0.9, n=4, gain=0.35)
    sc.add(Track("Radio presenter (dash centre)", sy.voice("radio"), [(0, (0, 0.3, 1.0))], kind="dialogue"), at=0.2)
    sc.add(Track("Nav chime", sy.chime(2), [(0, (-0.6, 0.3, 1.0))], gain=0.6), at=7.2)
    sc.add(Track("Navigation (front left)", sy.voice("nav"), [(0, (-0.6, 0.3, 1.0))], kind="dialogue"), at=8.0)
    sc.add(Track("Phone caller (front centre)", sy.voice("caller"), [(0, (0, 0.3, 1.0))], kind="dialogue"), at=11.0)
    return sc


def car_rain() -> Scene:
    sc = Scene("Rain and a passing truck", "Rain on the roof and glass, then a truck overtaking on your left "
               "from behind to the front, then a motorbike crossing right to left in front.", 16)
    sc.ambience("Rain on the roof", lambda k: sy.rain(16, seed=70 + k, intensity=1.6), (0, 1.3, 1.2), 0.8, n=6, gain=0.45)
    sc.add(Track("Truck overtaking (left)", sy.engine_passby(8),
                 [(0, (-3.0, 12, 1.0)), (4, (-3.0, 1.4, 1.0)), (8, (-3.0, -10, 1.0))], gain=0.9, lfe=0.5), at=2)
    sc.add(Track("Motorbike crossing in front", sy.engine_passby(5, seed=12),
                 [(0, (8, -6, 1.0)), (5, (-8, -6, 1.0))], gain=0.6), at=10)
    return sc


def car_movie() -> Scene:
    sc = Scene("Movie night in the car", "A film soundtrack played in the cabin: a voice, a jet flyover from the "
               "windshield to the rear window, and an explosion.", 14)
    sc.add(Track("Score", sy.pad(14, notes=(45, 52, 57, 61)), [(0, (0, 0.3, 1.0))], gain=0.35, kind="music"))
    sc.add(Track("Leo (front centre)", sy.voice("leo_1"), [(0, (0, 0.3, 1.0))], kind="dialogue"), at=0.3)
    sc.add(Track("Jet flyover", sy.jet(4), [(0, (0, -3, 1.4)), (2, (0, 1.4, 1.6)), (4, (0, 6, 1.4))], gain=0.9, lfe=0.3), at=5)
    sc.add(Track("Explosion ahead", sy.explosion(14, at=10.0), [(0, (-1.0, -4, 1.0))], gain=1.0, lfe=1.0))
    return sc


def car_test() -> Scene:
    sc = Scene("Calibration test", "Pink-noise bursts moving around the cabin, then a centred voice.", 12)
    path = [(0, (-0.7, 0.3, 1.0)), (2, (0, 0.3, 1.0)), (4, (0.7, 0.3, 1.0)), (6, (0.7, 2.2, 1.0)), (8, (-0.7, 2.2, 1.0))]
    sc.add(Track("Noise moving around", sy.pink_bursts(8), path, gain=0.7))
    sc.add(Track("Centre voice", sy.voice("test_voice"), [(0, (0, 0.3, 1.0))], kind="dialogue"), at=8.2)
    return sc


SCENES = {
    "cinema": {"Two characters talking": cinema_dialogue, "Forest at dawn": cinema_forest,
               "Storm and flyover": cinema_storm, "Calibration test": cinema_test},
    "car": {"Radio, navigation and a phone call": car_conversation, "Rain and a passing truck": car_rain,
            "Movie night in the car": car_movie, "Calibration test": car_test},
}
