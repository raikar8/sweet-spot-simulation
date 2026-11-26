"""Reference rendering algorithms. Read these as worked examples for your own."""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from ..geometry import SPEED_OF_SOUND as C, wrap180
from .base import Drive, RenderContext, Renderer, register


def _direction(p: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = p - ref
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([0.0, -1.0, 0.0])


@register
class VBAP(Renderer):
    """Vector Base Amplitude Panning (Pulkki 1997), 3-D with a triangulated speaker hull.

    Each channel is a direction seen from the mix reference point. The hull of those
    directions (plus virtual speakers straight up and down when the layout has no
    height) is split into triangles. A source direction p is rendered by the
    triangle whose three gains g = p^T L^-1 are all >= 0, normalised to constant
    power. Gains on virtual speakers are discarded, which reduces to classic
    pairwise 2-D VBAP for a flat layout.
    """

    name = "Channel panning (VBAP)"
    description = "Amplitude panning between the nearest channels. The industry default."

    def prepare(self, ctx: RenderContext) -> None:
        self.chans = list(ctx.channels)
        U = [ctx.channel_unit(c) for c in self.chans]
        virt = []
        z = np.array([u[2] for u in U])
        if z.max() < 0.5:
            virt.append(np.array([0, 0, 1.0]))
        if z.min() > -0.5:
            virt.append(np.array([0, 0, -1.0]))
        self.U = np.array(U + virt)
        self.n_real = len(U)
        if len(self.U) >= 4:
            hull = ConvexHull(self.U)
            self.tris = [t for t in hull.simplices]
            self.inv = []
            for t in self.tris:
                M = self.U[t]
                self.inv.append(np.linalg.inv(M) if abs(np.linalg.det(M)) > 1e-6 else None)
        else:
            self.tris, self.inv = [], []

    def channel_gains(self, p_unit: np.ndarray) -> dict:
        best = None
        for t, inv in zip(self.tris, self.inv):
            if inv is None:
                continue
            g = p_unit @ inv
            if g.min() >= -1e-6:
                best = (t, np.maximum(g, 0))
                break
        out = {}
        if best is None:  # outside every triangle (e.g. stereo, source behind): nearest channel
            k = int(np.argmax(self.U[: self.n_real] @ p_unit))
            return {self.chans[k]: 1.0}
        t, g = best
        g = np.where(t < self.n_real, g, 0.0)
        if g.sum() == 0:
            k = int(np.argmax(self.U[: self.n_real] @ p_unit))
            return {self.chans[k]: 1.0}
        g = g / np.linalg.norm(g)
        for idx, gi in zip(t, g):
            if idx < self.n_real and gi > 0:
                out[self.chans[idx]] = float(gi)
        return out

    def drive(self, source_pos, ctx):
        g = ctx.spread_channels(self.channel_gains(_direction(source_pos, ctx.ref)))
        return Drive(g, np.zeros(ctx.n))


@register
class Snap(VBAP):
    """Send the whole sound to the single nearest channel (Atmos-style 'snap')."""

    name = "Snap to nearest speaker"
    description = "Every sound comes from one real speaker: sharp from every seat, but moves in steps."

    def drive(self, source_pos, ctx):
        p = _direction(source_pos, ctx.ref)
        k = int(np.argmax(self.U[: self.n_real] @ p))
        return Drive(ctx.spread_channels({self.chans[k]: 1.0}), np.zeros(ctx.n))


@register
class FirstOrderAmbisonics(Renderer):
    """First-order Ambisonics, encoded and decoded in one step with an in-phase (cardioid) decoder.

    Channel c with direction u_c gets g_c = max(0, (1 + u_c . p) / 2), normalised to
    constant power. Every channel on the source's side plays part of the sound.
    """

    name = "First-order Ambisonics"
    description = "Every speaker plays a share. Smooth and enveloping, but vague and wide."

    def prepare(self, ctx):
        self.chans = list(ctx.channels)
        self.U = np.array([ctx.channel_unit(c) for c in self.chans])

    def drive(self, source_pos, ctx):
        p = _direction(source_pos, ctx.ref)
        g = np.maximum(0.0, 0.5 + 0.5 * (self.U @ p))
        g = g / (np.linalg.norm(g) + 1e-12)
        return Drive(ctx.spread_channels(dict(zip(self.chans, g))), np.zeros(ctx.n))


@register
class WFS(Renderer):
    """Wave field synthesis style point source (2.5-D, simplified).

    The virtual source S sits `push` metres behind the intended position, outside
    the speaker line. Every speaker i that radiates away from S (n_i . (x_i - S) > 0)
    plays the signal delayed by |x_i - S| / c with gain cos(phi) / sqrt(|x_i - S|),
    so together they approximate the wavefront S would emit. It needs speakers
    every 10 to 20 cm; with fewer, expect spatial aliasing (colouration).
    """

    name = "Wave field synthesis (point source)"
    description = "Rebuilds the wavefront of a virtual source. Holds position across seats if speakers are dense."
    uses_calibration = False

    def drive(self, source_pos, ctx):
        push = 0.25 if ctx.room.id == "car" else 2.0
        u = _direction(source_pos, ctx.ref)
        S = source_pos + push * u
        g = np.zeros(ctx.n)
        d = np.zeros(ctx.n)
        for i, s in enumerate(ctx.speakers):
            if s.is_sub:
                continue
            v = s.pos - S
            r = max(0.2, float(np.linalg.norm(v)))
            c = float(np.dot(s.axis, v / r))
            if c > 0:
                g[i] = c / np.sqrt(r)
                d[i] = r / C
        if g.sum() == 0:  # no speaker faces away from S: fall back to the nearest channel
            fb = Snap()
            fb.prepare(ctx)
            return fb.drive(source_pos, ctx)
        g = g / np.linalg.norm(g)
        d = np.where(g > 0, d - d[g > 0].min(), 0.0)
        return Drive(g, d)


REFERENCE = ["VBAP", "Snap", "FirstOrderAmbisonics", "WFS"]
