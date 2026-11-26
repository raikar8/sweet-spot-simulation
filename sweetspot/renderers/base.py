"""The interface every rendering algorithm implements.

A renderer answers one question, many times per second of audio:

    "A sound should appear at position P in the room.
     What gain and what extra delay should each loudspeaker get?"

Write your own by subclassing Renderer and implementing drive(). Put the file in
user_renderers/ (or sweetspot/renderers/) and decorate the class with
@register: it then shows up in the app and in the evals. See
user_renderers/my_renderer.py for a commented template.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Room, Speaker, azimuth_from


@dataclass
class Drive:
    """What each loudspeaker plays for one source at one instant.

    gains:  linear gain per speaker (same order as ctx.speakers). The engine adds
            the subwoofer feed itself (bass management), so leave sub gains at 0.
    delays: extra delay per speaker in seconds (>= 0), e.g. for wave field synthesis.
    """
    gains: np.ndarray
    delays: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "Drive":
        return cls(np.zeros(n), np.zeros(n))


@dataclass
class RenderContext:
    """Everything a renderer may look at. Built once per room + layout."""
    room: Room
    speakers: list[Speaker]
    ref: np.ndarray                       # the mix reference point (x, y, z)
    channels: list[str] = field(default_factory=list)          # main channels, no LFE
    channel_speakers: dict = field(default_factory=dict)        # channel -> [speaker indices]

    @classmethod
    def build(cls, room: Room, speakers: list[Speaker]) -> "RenderContext":
        ch_spk: dict = {}
        for i, s in enumerate(speakers):
            if s.is_sub:
                continue
            ch_spk.setdefault(s.channel, []).append(i)
        return cls(room, speakers, room.ref3(), list(ch_spk), ch_spk)

    @property
    def n(self) -> int:
        return len(self.speakers)

    def channel_position(self, ch: str) -> np.ndarray:
        """Centroid of a channel's speakers (a surround array's middle)."""
        return np.mean([self.speakers[i].pos for i in self.channel_speakers[ch]], axis=0)

    def channel_azimuth(self, ch: str) -> float:
        """Azimuth of a channel seen from the mix reference point (deg, + = right)."""
        return azimuth_from(self.ref, self.channel_position(ch))

    def channel_unit(self, ch: str) -> np.ndarray:
        d = self.channel_position(ch) - self.ref
        return d / np.linalg.norm(d)

    def spread_channels(self, ch_gains: dict) -> np.ndarray:
        """Turn per-channel gains into per-speaker gains (every speaker of a channel plays it)."""
        g = np.zeros(self.n)
        for ch, v in ch_gains.items():
            for i in self.channel_speakers.get(ch, []):
                g[i] = v
        return g


class Renderer:
    """Base class. Override drive(); optionally prepare()."""

    name: str = "Unnamed renderer"
    description: str = ""
    # If True the engine applies the room calibration (time alignment and level trim)
    # on top of your gains and delays. Set False if your algorithm controls timing itself.
    uses_calibration: bool = True

    def prepare(self, ctx: RenderContext) -> None:
        """Called once when the room or layout changes. Precompute here."""

    def drive(self, source_pos: np.ndarray, ctx: RenderContext) -> Drive:
        raise NotImplementedError


REGISTRY: dict[str, type[Renderer]] = {}


def register(cls: type[Renderer]) -> type[Renderer]:
    """Class decorator: make a renderer available to the app and the evals."""
    REGISTRY[cls.__name__] = cls
    return cls
