"""Basic geometry: rooms, loudspeakers, listeners.

Coordinate system (metres), used everywhere in this package:
    x  to the right when facing the screen / windshield
    y  towards the back of the room (the screen or dashboard is at small y)
    z  up from the floor
A listener faces -y (towards the screen) unless their yaw is changed.
Azimuths are in degrees, 0 = straight ahead, positive = to the RIGHT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at about 20 C


def vec(*xyz) -> np.ndarray:
    return np.asarray(xyz, dtype=float)


@dataclass
class Speaker:
    """One physical loudspeaker.

    channel: the mix channel it belongs to ("L", "Lss", "FL", ...). Several
             speakers can share a channel (a surround array).
    band:    "full", "high" (tweeter), "low" (woofer) or "sub".
    aim:     a point in the room the speaker is pointed at.
    """
    name: str
    channel: str
    pos: np.ndarray
    aim: np.ndarray
    band: str = "full"

    @property
    def axis(self) -> np.ndarray:
        d = self.aim - self.pos
        return d / (np.linalg.norm(d) + 1e-12)

    @property
    def is_sub(self) -> bool:
        return self.band == "sub"


@dataclass
class Surface:
    """Reflection coefficient (pressure, 0..1) of each of the six walls of the box."""
    x0: float = 0.7
    x1: float = 0.7
    y0: float = 0.7  # front wall (screen / dashboard)
    y1: float = 0.7  # back wall
    z0: float = 0.6  # floor
    z1: float = 0.7  # ceiling


@dataclass
class Room:
    """A shoebox room with seats, speaker layouts and calibration points."""
    id: str
    label: str
    x0: float
    x1: float
    y0: float
    y1: float
    height: float
    ear_height: Callable[[float, float], float]
    seats: list
    named_seats: dict
    ref: np.ndarray                 # the point the mix is panned around (x, y)
    cal_points: dict                # name -> (x, y)
    layouts: dict                   # name -> (label, builder)
    default_layout: str
    rt60: float                     # reverberation time, s
    walls: Surface = field(default_factory=Surface)
    reflection_order: int = 2
    grid_step: float = 1.0          # spacing of the eval grid, m

    def listener(self, x: float, y: float, yaw_deg: float = 0.0) -> "Listener":
        return Listener(vec(x, y, self.ear_height(x, y)), yaw_deg)

    def build_layout(self, name: Optional[str] = None) -> list:
        return self.layouts[name or self.default_layout][1]()

    def ref3(self) -> np.ndarray:
        return vec(self.ref[0], self.ref[1], self.ear_height(*self.ref))

    def cal3(self, name: str) -> np.ndarray:
        x, y = self.cal_points[name]
        return vec(x, y, self.ear_height(x, y))

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class Listener:
    pos: np.ndarray
    yaw_deg: float = 0.0  # positive = head turned to the right

    def to_head(self, p: np.ndarray) -> tuple[float, float, float]:
        """Direction of point p seen from this head: (azimuth right+, elevation, distance)."""
        d = p - self.pos
        r = float(np.linalg.norm(d)) + 1e-12
        az_room = np.degrees(np.arctan2(d[0], -d[1]))
        az = (az_room - self.yaw_deg + 180.0) % 360.0 - 180.0
        el = float(np.degrees(np.arcsin(np.clip(d[2] / r, -1, 1))))
        return float(az), el, r


def azimuth_from(origin_xy, p) -> float:
    """Room azimuth (degrees, 0 = towards the screen, + = right) of p seen from origin."""
    return float(np.degrees(np.arctan2(p[0] - origin_xy[0], -(p[1] - origin_xy[1]))))


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0
