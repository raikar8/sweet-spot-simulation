import numpy as np
import pytest

from sweetspot.acoustics import BRIROptions, calibrate, direct_info, image_sources
from sweetspot.engine import Seat, SystemSetup
from sweetspot.evals import metrics as M
from sweetspot.geometry import SPEED_OF_SOUND as C
from sweetspot.hrtf import HRTF
from sweetspot.renderers import REGISTRY, RenderContext, load_user_renderers
from sweetspot.rooms import CAR, CINEMA


@pytest.fixture(scope="module")
def hrtf():
    return HRTF.kemar()


def test_hrtf_convention(hrtf):
    h = hrtf.nearest(90, 0)  # source on the right
    assert np.sum(h[:, 1] ** 2) > 4 * np.sum(h[:, 0] ** 2)


def test_lateral_estimator_matches_hrtf(hrtf):
    for az in (-60, -30, 0, 30, 60):
        assert abs(M.lateral_angle(hrtf.nearest(az, 0).astype(float), hrtf) - az) <= 5


def test_image_sources_first_order():
    spk = CINEMA.build_layout("7.1")[0]
    paths = image_sources(CINEMA, spk, 1)
    assert len(paths) == 7
    xs = sorted(p.pos[0] for p in paths if p.order == 1 and p.pos[1] == spk.pos[1] and p.pos[2] == spk.pos[2])
    assert xs == pytest.approx([2 * CINEMA.x0 - spk.pos[0], 2 * CINEMA.x1 - spk.pos[0]])


def test_calibration_aligns_arrivals():
    spk = CINEMA.build_layout("7.1")
    q = CINEMA.ref3()
    cal = calibrate(spk, q)
    t = [direct_info(s, q)[0] / C + d for s, d in zip(spk, cal.delays)]
    assert np.ptp(t) < 1e-9


@pytest.mark.parametrize("room", [CINEMA, CAR])
def test_vbap_power_and_on_channel(room):
    spk = room.build_layout()
    ctx = RenderContext.build(room, spk)
    r = REGISTRY["VBAP"]()
    r.prepare(ctx)
    for ch in ctx.channels:
        g = r.channel_gains(ctx.channel_unit(ch))
        assert g.get(ch, 0) == pytest.approx(1.0, abs=1e-6)
    for az in np.linspace(-180, 180, 37):
        p = ctx.ref + np.array([np.sin(np.radians(az)), -np.cos(np.radians(az)), 0]) * 5
        g = np.array(list(r.channel_gains((p - ctx.ref) / 5).values()))
        assert np.sum(g ** 2) == pytest.approx(1.0, abs=1e-6)


def test_wfs_delays_non_negative():
    spk = CINEMA.build_layout("dense")
    ctx = RenderContext.build(CINEMA, spk)
    d = REGISTRY["WFS"]().drive(np.array([-3.0, 0.6, 4.0]), ctx)
    assert d.delays.min() >= 0 and d.gains.max() > 0


def test_reference_seat_centre_dialogue_is_centred(hrtf):
    seat = Seat(SystemSetup(CINEMA, "7.1", "Reference seat", brir=BRIROptions(late_tail=False)), CINEMA.listener(0, 19), hrtf)
    r = REGISTRY["VBAP"]()
    r.prepare(seat.ctx)
    h = seat.system_ir(r, np.array([0.0, 0.6, 4.0]))
    assert abs(M.lateral_angle(h, hrtf)) <= 5


def test_user_template_loads():
    assert load_user_renderers() == []
    assert "DistanceWeighted" in REGISTRY


def test_every_renderer_drives_every_layout():
    load_user_renderers()
    for room in (CINEMA, CAR):
        for lay in room.layouts:
            ctx = RenderContext.build(room, room.build_layout(lay))
            for name, cls in REGISTRY.items():
                r = cls()
                r.prepare(ctx)
                d = r.drive(ctx.ref + np.array([1.0, -3.0, 0.0]), ctx)
                assert d.gains.shape == (ctx.n,) and np.isfinite(d.gains).all(), (room.id, lay, name)
