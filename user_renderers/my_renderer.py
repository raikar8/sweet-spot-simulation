"""Template: write your own rendering algorithm here.

Every .py file in this folder is loaded by the app and the evals. A class
decorated with @register appears in the renderer lists automatically.

Your job: given where a sound should be (source_pos, metres, room coordinates)
return a gain and an extra delay for every loudspeaker.

Useful things on ctx (a RenderContext):
    ctx.speakers                 list of Speaker (name, channel, pos, aim, band, axis)
    ctx.channels                 main channel names, e.g. ["L", "C", "R", "Lss", ...]
    ctx.channel_speakers[ch]     indices of the speakers in a channel (arrays have many)
    ctx.channel_azimuth(ch)      channel direction seen from the mix reference point (deg, + = right)
    ctx.channel_unit(ch)         the same as a unit vector
    ctx.ref                      the mix reference point
    ctx.room                     the Room (size, seats, rt60, ...)
    ctx.spread_channels({ch: g}) per-channel gains -> per-speaker gain array

Coordinates: x right, y towards the back (screen at y = 0), z up.

Test it:  python -m sweetspot.evals.run --renderers DistanceWeighted VBAP --room car
"""
import numpy as np

from sweetspot.renderers import Drive, Renderer, register


@register
class DistanceWeighted(Renderer):
    """Example: weight every channel by how close it is to the source (a 'distance-based' panner, DBAP-like)."""

    name = "Example: distance-based panning (DBAP-like)"
    description = "Template algorithm. Copy this file and change drive()."
    uses_calibration = True      # let the room's time alignment and level trim apply

    rolloff_db = 6.0             # attenuation per doubling of distance

    def prepare(self, ctx):
        self.chans = list(ctx.channels)
        self.pos = np.array([ctx.channel_position(c) for c in self.chans])

    def drive(self, source_pos, ctx):
        d = np.linalg.norm(self.pos - source_pos, axis=1) + 0.5   # blur radius 0.5 m
        a = self.rolloff_db / (20 * np.log10(2))
        g = 1.0 / d ** a
        g /= np.linalg.norm(g)                                   # constant power
        gains = ctx.spread_channels(dict(zip(self.chans, g)))
        return Drive(gains, np.zeros(ctx.n))
