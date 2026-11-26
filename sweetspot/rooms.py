"""Room and loudspeaker-layout definitions.

Dimensions and speaker positions are typical values, not measurements of a
specific cinema or car. Edit or add your own here.
"""
from __future__ import annotations

import numpy as np

from .geometry import Room, Speaker, Surface, vec


# ---------------------------------------------------------------- cinema
def _cinema_ear(x, y):
    # stadium seating: rows 2 m apart rise 0.35 m each, first row at y = 7
    return 1.2 + float(np.clip(y - 7.0, 0, 18)) * 0.175


def _cinema(kind: str):
    S = []

    def add(name, ch, x, y, z, ax, ay, az=1.5, band="full"):
        S.append(Speaker(name, ch, vec(x, y, z), vec(ax, ay, az), band))

    if kind == "dense":
        n = 0
        def ad(x, y, ax, ay):
            nonlocal n
            n += 1
            add(f"S{n}", f"S{n}", x, y, 4.0, ax, ay)
        for x in np.arange(-7.5, 7.51, 1.5):
            ad(x, 0.6, x, 19)
        for y in np.arange(3, 27.1, 1.5):
            ad(8.8, y, -8.8, y)
        for x in np.arange(7.5, -7.51, -1.5):
            ad(x, 27.8, x, 0)
        for y in np.arange(27, 2.9, -1.5):
            ad(-8.8, y, 8.8, y)
        add("Sub", "LFE", 0, 0.4, 1.0, 0, 19, band="sub")
        return S

    add("L", "L", -6, 0.6, 4.0, 0, 19)
    if kind != "stereo":
        add("C", "C", 0, 0.6, 4.0, 0, 19)
    add("R", "R", 6, 0.6, 4.0, 0, 19)
    add("Sub", "LFE", 0, 0.4, 1.0, 0, 19, band="sub")
    if kind == "stereo":
        return S
    side_l = "Lss" if kind in ("7.1", "7.1.4") else "Ls"
    side_r = "Rss" if kind in ("7.1", "7.1.4") else "Rs"
    rear_l = "Lrs" if kind in ("7.1", "7.1.4") else "Ls"
    rear_r = "Rrs" if kind in ("7.1", "7.1.4") else "Rs"
    for i, y in enumerate([9, 13, 17, 21, 25]):
        add(f"Ls{i+1}", side_l, -8.8, y, 5.5, 3, y + 1)
        add(f"Rs{i+1}", side_r, 8.8, y, 5.5, -3, y + 1)
    for i, x in enumerate([-6, -2]):
        add(f"Lr{i+1}", rear_l, x, 27.8, 5.5, x * 0.4, 14)
    for i, x in enumerate([2, 6]):
        add(f"Rr{i+1}", rear_r, x, 27.8, 5.5, x * 0.4, 14)
    if kind == "7.1.4":
        for nm, x, y in [("Ltf", -4, 11), ("Rtf", 4, 11), ("Ltr", -4, 22), ("Rtr", 4, 22)]:
            add(nm, nm, x, y, 9.5, x, y, az=0.0)
    return S


CINEMA = Room(
    id="cinema", label="Cinema (about 300 seats)",
    x0=-9, x1=9, y0=0, y1=28, height=10.0,
    ear_height=lambda x, y: _cinema_ear(x, y),
    seats=[(float(x), float(y)) for y in range(7, 26, 2) for x in range(-7, 8)],
    named_seats={
        "Reference seat": (0, 19), "Front row centre": (0, 7), "Front row far left": (-7, 7),
        "Middle right": (6, 15), "Back row left": (-6, 25),
    },
    ref=np.array([0.0, 19.0]),
    cal_points={"Reference seat": (0, 19)},
    layouts={
        "7.1": ("7.1: screen L C R, side + rear surround arrays, sub", lambda: _cinema("7.1")),
        "7.1.4": ("7.1.4: as 7.1 plus four ceiling speakers", lambda: _cinema("7.1.4")),
        "5.1": ("5.1: screen L C R, two surround arrays, sub", lambda: _cinema("5.1")),
        "stereo": ("Stereo: screen L R, sub", lambda: _cinema("stereo")),
        "dense": ("Dense ring: a speaker every 1.5 m, each driven separately", lambda: _cinema("dense")),
    },
    default_layout="7.1",
    rt60=0.6,
    walls=Surface(x0=0.55, x1=0.55, y0=0.75, y1=0.45, z0=0.5, z1=0.6),
    reflection_order=2,
    grid_step=1.0,
)


# ---------------------------------------------------------------- car
def _car(kind: str):
    S = []

    def add(name, ch, x, y, z, ax, ay, az=1.1, band="full"):
        S.append(Speaker(name, ch, vec(x, y, z), vec(ax, ay, az), band))

    if kind == "basic":
        add("FL door", "FL", -0.74, 0.8, 0.45, 0.4, 0.95)
        add("FR door", "FR", 0.74, 0.8, 0.45, -0.4, 0.95)
        add("RL door", "RL", -0.74, 1.75, 0.45, 0.4, 1.9)
        add("RR door", "RR", 0.74, 1.75, 0.45, -0.4, 1.9)
        return S
    add("FL tweeter", "FL", -0.62, 0.25, 1.0, 0.37, 0.95, band="high")
    add("FL woofer", "FL", -0.74, 0.8, 0.45, 0.4, 0.95, band="low")
    add("Centre", "C", 0.0, 0.3, 0.95, 0.0, 1.4)
    add("FR tweeter", "FR", 0.62, 0.25, 1.0, -0.37, 0.95, band="high")
    add("FR woofer", "FR", 0.74, 0.8, 0.45, -0.4, 0.95, band="low")
    add("RL door", "RL", -0.74, 1.75, 0.45, 0.4, 1.9)
    add("RR door", "RR", 0.74, 1.75, 0.45, -0.4, 1.9)
    add("Sub", "LFE", 0.0, 2.38, 0.55, 0.0, 1.4, band="sub")
    return S


CAR = Room(
    id="car", label="Car cabin (mid-size)",
    x0=-0.75, x1=0.75, y0=0.0, y1=2.45, height=1.25,
    ear_height=lambda x, y: 1.1,
    seats=[(-0.37, 0.95), (0.37, 0.95), (-0.37, 1.9), (0.0, 1.9), (0.37, 1.9)],
    named_seats={"Front left": (-0.37, 0.95), "Front right": (0.37, 0.95),
                 "Rear left": (-0.37, 1.9), "Rear right": (0.37, 1.9), "Cabin centre": (0.0, 1.4)},
    ref=np.array([0.0, 1.4]),
    cal_points={"Front left": (-0.37, 0.95), "Front right": (0.37, 0.95), "Cabin centre": (0.0, 1.4)},
    layouts={
        "premium": ("Premium: 2-way fronts, centre, rear doors, sub", lambda: _car("premium")),
        "basic": ("Basic: four door speakers", lambda: _car("basic")),
    },
    default_layout="premium",
    rt60=0.08,
    # glass is very reflective, seats and headliner absorb
    walls=Surface(x0=0.85, x1=0.85, y0=0.9, y1=0.6, z0=0.4, z1=0.5),
    reflection_order=3,
    grid_step=0.1,
)

ROOMS = {"cinema": CINEMA, "car": CAR}


def eval_grid(room: Room) -> list[tuple[float, float]]:
    """Listening positions used for sweet-spot maps."""
    if room.id == "car":
        xs = np.arange(-0.55, 0.551, room.grid_step)
        ys = np.arange(0.8, 2.11, room.grid_step)
    else:
        xs = np.arange(-7.5, 7.51, room.grid_step)
        ys = np.arange(6.5, 25.51, room.grid_step)
    return [(float(x), float(y)) for y in ys for x in xs]
