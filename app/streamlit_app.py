"""Sweet Spot Lab: listen to cinema and car speaker systems from any seat, compare algorithms, run evals.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from sweetspot.acoustics import BRIROptions
from sweetspot.engine import Seat, SystemSetup
from sweetspot.evals import run as ev
from sweetspot.hrtf import FS, HRTF
from sweetspot.renderers import REGISTRY, load_user_renderers
from sweetspot.rooms import ROOMS
from sweetspot.scenes import SCENES, Scene, Track

st.set_page_config(page_title="Sweet Spot Lab", page_icon="🎧", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1400px}
  h1 {font-weight: 800; letter-spacing: -0.01em}
  .ab-card {border: 1px solid rgba(128,128,128,.35); border-radius: 12px; padding: 12px 14px 4px}
  .ab-tag {font-weight: 800; font-size: 1.4rem; margin-right: .4rem}
  .muted {opacity: .7; font-size: .9rem}
</style>""", unsafe_allow_html=True)

USER_ERRORS = load_user_renderers()
CH_COLORS = ["#d9622b", "#7a5af0", "#2a8fd6", "#1f9e77", "#c7489a", "#9a7b12", "#4f7f9e", "#e0a030",
             "#5bb5a2", "#b05f3c", "#8d6cd9", "#3e8e41"]


# ---------------------------------------------------------------- cached heavy work
@st.cache_resource(show_spinner=False)
def get_hrtf(sofa_bytes: bytes | None, name: str):
    if sofa_bytes is None:
        return HRTF.kemar()
    with tempfile.NamedTemporaryFile(suffix=".sofa", delete=False) as f:
        f.write(sofa_bytes)
    h = HRTF.from_sofa(f.name)
    h.name = name
    return h


@st.cache_resource(show_spinner=False, max_entries=16)
def get_seat(room_id, layout, x, y, yaw, cal, align, level, refl, head, _hrtf, hrtf_name):
    room = ROOMS[room_id]
    setup = SystemSetup(room, layout, cal, align, level, BRIROptions(reflections=refl, head_model=head))
    return Seat(setup, room.listener(x, y, yaw), _hrtf)


@st.cache_resource(show_spinner=False, max_entries=8)
def get_scene(room_id, name, upload_key):
    if name == "My own audio file":
        data, (x, y, z) = st.session_state["upload"]
        sc = Scene("My own audio file", "Your WAV, placed where you chose.", min(30.0, len(data) / FS + 0.5))
        sc.add(Track("Your sound", data, [(0, (x, y, z))]))
        return sc
    return SCENES[room_id][name]()


def wav_bytes(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, FS, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def render(seat_args, scene_args, renderer_name):
    key = (seat_args[:10] + seat_args[11:], scene_args, renderer_name)
    cache = st.session_state.setdefault("renders", {})
    if key not in cache:
        seat = get_seat(*seat_args)
        scene = get_scene(*scene_args)
        cache[key] = seat.render(scene, REGISTRY[renderer_name]())
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
    return cache[key]


def loudness_match(a, b):
    """Same RMS for A and B (fair A/B listening), then one common peak limit."""
    ra = np.sqrt(np.mean(a ** 2)) + 1e-12
    rb = np.sqrt(np.mean(b ** 2)) + 1e-12
    b = b * ra / rb
    k = min(1.0, 0.89 / max(np.abs(a).max(), np.abs(b).max()))
    return a * k, b * k, 20 * np.log10(ra / rb)


# ---------------------------------------------------------------- sidebar: the system
with st.sidebar:
    st.header("System")
    room_id = st.radio("Room", list(ROOMS), format_func=lambda r: ROOMS[r].label, horizontal=True)
    room = ROOMS[room_id]
    layout = st.selectbox("Speaker layout", list(room.layouts), format_func=lambda k: room.layouts[k][0])

    if st.session_state.get("room_seen") != room_id:
        st.session_state["room_seen"] = room_id
        st.session_state["seat"] = tuple(next(iter(room.named_seats.values())))
    names = list(room.named_seats)
    pick = st.selectbox("Jump to a seat", ["(keep current)"] + names)
    if pick != "(keep current)" and st.session_state.get("last_pick") != (room_id, pick):
        st.session_state["seat"] = tuple(room.named_seats[pick])
    st.session_state["last_pick"] = (room_id, pick)
    sx, sy_ = st.session_state["seat"]
    st.caption(f"Your seat: x = {sx:.2f} m, y = {sy_:.2f} m. Click any seat on the map to move.")
    yaw = st.slider("Turn your head (deg, + = right)", -90, 90, 0, 5)

    st.subheader("Calibration")
    cal_opts = list(room.cal_points) + ["seat", None]
    cal = st.selectbox("Tuned for", cal_opts, format_func=lambda c: "My seat (follows me)" if c == "seat" else "Not tuned" if c is None else c)
    align = st.checkbox("Time alignment", True, disabled=cal is None)
    level = st.checkbox("Level calibration", True, disabled=cal is None)
    if cal is None:
        align = level = False
        cal = next(iter(room.cal_points))

    st.subheader("Acoustics and head")
    refl = st.checkbox("Wall reflections (image sources) and reverb", True)
    head = st.radio("Headphone rendering", ["hrtf", "stereo"], horizontal=True,
                    format_func=lambda h: "Binaural (HRTF)" if h == "hrtf" else "Plain stereo")
    sofa = st.file_uploader("Your own HRTF (.sofa), optional", type=["sofa"])
    hrtf = get_hrtf(sofa.getvalue() if sofa else None, sofa.name if sofa else "MIT KEMAR")
    st.caption(f"HRTF: {hrtf.name}")

seat_args = (room_id, layout, sx, sy_, yaw, cal, align, level, refl, head, hrtf, hrtf.name)
seat = get_seat(*seat_args)

st.title("Sweet Spot Lab")
st.markdown('<span class="muted">Hear a cinema or a car from any seat, on headphones. Compare rendering algorithms, '
            'write your own, and measure where each one works.</span>', unsafe_allow_html=True)
if USER_ERRORS:
    st.error("Some of your renderers failed to load:\n\n" + "\n\n".join(USER_ERRORS))

tab_listen, tab_eval, tab_write, tab_how = st.tabs(["🎧 Listen and compare", "📊 Sweet-spot evals", "✍️ Write your algorithm", "ℹ️ How it works"])


# ---------------------------------------------------------------- map
def room_map(seat: Seat, scene: Scene | None, height=560):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=room.x0, x1=room.x1, y0=room.y0, y1=room.y1, line=dict(color="gray", width=2),
                  fillcolor="rgba(128,128,128,0.06)")
    if room.id == "cinema":
        fig.add_shape(type="line", x0=-7.5, x1=7.5, y0=0.2, y1=0.2, line=dict(color="gray", width=6))
        fig.add_annotation(x=0, y=-0.9, text="SCREEN", showarrow=False, font=dict(size=10))
    else:
        fig.add_annotation(x=0, y=-0.12, text="WINDSHIELD", showarrow=False, font=dict(size=10))
    sx_, sy2 = zip(*room.seats)
    fig.add_trace(go.Scatter(x=sx_, y=sy2, mode="markers", name="Seats", marker=dict(size=9 if room.id == "car" else 7,
                  color="rgba(128,128,128,0.45)", symbol="square"), hovertemplate="seat (%{x}, %{y})<extra></extra>"))
    chans = list(dict.fromkeys(s.channel for s in seat.speakers))
    for k, ch in enumerate(chans):
        sp = [s for s in seat.speakers if s.channel == ch]
        fig.add_trace(go.Scatter(x=[s.pos[0] for s in sp], y=[s.pos[1] for s in sp], mode="markers",
                                 name=ch if len(chans) <= 12 else "speaker", showlegend=len(chans) <= 12,
                                 marker=dict(size=13, color=CH_COLORS[k % len(CH_COLORS)], symbol="triangle-up",
                                             angle=[float(np.degrees(np.arctan2(s.axis[0], -s.axis[1]))) for s in sp],
                                             line=dict(width=1, color="black")),
                                 text=[s.name for s in sp], hovertemplate="%{text}<extra></extra>"))
    if scene is not None:
        for tr in scene.tracks:
            if tr.kind == "ambience":
                continue
            ps = np.array([k[1] for k in tr.keys])
            fig.add_trace(go.Scatter(x=ps[:, 0], y=ps[:, 1], mode="lines+markers", name=tr.name, showlegend=False,
                                     line=dict(dash="dot", width=1.5, color="#888"), marker=dict(size=7, symbol="star", color="#444"),
                                     hovertemplate=tr.name + "<extra></extra>"))
    cp = seat.cal.point
    fig.add_trace(go.Scatter(x=[cp[0]], y=[cp[1]], mode="markers", name="Calibrated here",
                             marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="#d9622b", width=2, dash="dot"))))
    fig.add_trace(go.Scatter(x=[sx], y=[sy_], mode="markers", name="You",
                             marker=dict(size=16, color="#1f5fd1", line=dict(color="white", width=2))))
    pad = 0.3 if room.id == "car" else 2.5
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="v", x=1.02, y=1, font=dict(size=11)),
                      xaxis=dict(range=[room.x0 - pad, room.x1 + pad], scaleanchor="y", title=None),
                      yaxis=dict(range=[room.y1 + pad, room.y0 - pad * 1.5], title="metres from the front"),
                      clickmode="event+select", dragmode="pan")
    return fig


# ---------------------------------------------------------------- listen tab
with tab_listen:
    left, right = st.columns([1.05, 1], gap="large")
    with right:
        scene_names = list(SCENES[room_id]) + ["My own audio file"]
        scene_name = st.selectbox("Scene", scene_names)
        upload_key = None
        if scene_name == "My own audio file":
            up = st.file_uploader("WAV file", type=["wav", "flac", "ogg"])
            c1, c2, c3 = st.columns(3)
            ux = c1.number_input("x (m)", value=0.0, step=0.5 if room.id == "cinema" else 0.1)
            uy = c2.number_input("y (m)", value=float(room.y0 + 0.6 if room.id == "cinema" else 0.3), step=0.5 if room.id == "cinema" else 0.1)
            uz = c3.number_input("z (m)", value=4.0 if room.id == "cinema" else 1.0, step=0.5)
            if up is None:
                st.info("Upload a sound to place it in the room.")
                st.stop()
            data, fs = sf.read(io.BytesIO(up.getvalue()), always_2d=True)
            data = data.mean(axis=1).astype(np.float32)
            if fs != FS:
                from scipy.signal import resample_poly
                data = resample_poly(data, FS, fs).astype(np.float32)
            data = data[: 30 * FS] / (np.abs(data).max() + 1e-9) * 0.6
            st.session_state["upload"] = (data, (ux, uy, uz))
            upload_key = (up.name, up.size, ux, uy, uz)
        scene = get_scene(room_id, scene_name, upload_key)
        st.caption(scene.description)

        names = list(REGISTRY)
        fmt = lambda n: REGISTRY[n].name
        st.markdown("#### Compare two algorithms")
        ca, cb = st.columns(2)
        ra = ca.selectbox("A", names, index=names.index("VBAP"), format_func=fmt)
        rb = cb.selectbox("B", names, index=names.index("WFS") if "WFS" in names else 0, format_func=fmt)
        match = st.toggle("Match loudness of A and B (fair comparison)", True)
        if st.button("▶ Render A and B from my seat", type="primary", width="stretch"):
            st.session_state["want"] = True
        if st.session_state.get("want"):
            with st.spinner("Rendering both versions through every speaker, the room and your head…"):
                ya = render(seat_args, (room_id, scene_name, upload_key), ra)
                yb = render(seat_args, (room_id, scene_name, upload_key), rb)
            if match:
                ya, yb, diff = loudness_match(ya, yb)
            else:
                k = min(1.0, 0.89 / max(np.abs(ya).max(), np.abs(yb).max()))
                ya, yb, diff = ya * k, yb * k, 0.0
            for tag, nm, y in (("A", ra, ya), ("B", rb, yb)):
                st.markdown(f'<div class="ab-card"><span class="ab-tag">{tag}</span><b>{REGISTRY[nm].name}</b><br>'
                            f'<span class="muted">{REGISTRY[nm].description}</span></div>', unsafe_allow_html=True)
                st.audio(wav_bytes(y), format="audio/wav")
            if match and abs(diff) > 0.5:
                st.caption(f"B was {abs(diff):.1f} dB {'quieter' if diff > 0 else 'louder'} than A before matching.")
            st.caption("Use headphones. Change the seat, head turn or calibration and render again to hear what changes.")

    with left:
        ev_ = st.plotly_chart(room_map(seat, scene), width="stretch", on_select="rerun",
                              selection_mode="points", key=f"map_{room_id}")
        pts = (ev_ or {}).get("selection", {}).get("points", []) if isinstance(ev_, dict) else getattr(getattr(ev_, "selection", None), "points", [])
        if pts:
            p = pts[0]
            new = (float(p["x"]), float(p["y"]))
            if room.contains(*new) and new != st.session_state["seat"] and p.get("curve_number", 0) == 0:
                st.session_state["seat"] = new
                st.rerun()

        st.markdown("#### What reaches your seat")
        probe_tracks = [t for t in scene.tracks if t.kind != "ambience"]
        if probe_tracks:
            tname = st.selectbox("For the sound", [t.name for t in probe_tracks])
            tr = next(t for t in probe_tracks if t.name == tname)
            r = REGISTRY[ra]()
            r.prepare(seat.ctx)
            rows = seat.arrivals(r, tr.position(0))
            df = pd.DataFrame(rows)
            amax = df.amplitude.max() or 1
            df = df[df.amplitude > 0].copy()
            df["level_db"] = 20 * np.log10(df.amplitude / amax)
            df = df.sort_values("arrival_ms")[["speaker", "channel", "distance_m", "added_delay_ms", "arrival_ms", "level_db", "off_axis_deg"]]
            st.dataframe(df.round(2), hide_index=True, width="stretch", height=260)
            st.caption(f"With algorithm A. Arrivals within about 1 ms fuse into one image; later ones are "
                       f"dominated by the first (precedence effect). Spread here: {df.arrival_ms.max() - df.arrival_ms.min():.2f} ms.")


# ---------------------------------------------------------------- evals tab
with tab_eval:
    st.markdown("Measure every algorithm at every seat. For each seat and each probe sound the lab builds what "
                "reaches the two ears and computes the metrics below. The current room, layout and calibration settings apply.")
    c1, c2, c3 = st.columns([2, 1, 1])
    sel = c1.multiselect("Algorithms", list(REGISTRY), default=list(REGISTRY), format_func=lambda n: REGISTRY[n].name)
    step = c2.select_slider("Seat grid spacing (m)", [2.0, 1.5, 1.0] if room.id == "cinema" else [0.2, 0.15, 0.1], value=2.0 if room.id == "cinema" else 0.15)
    go_ = c3.button("Run evals", type="primary", width="stretch")
    if go_ and sel:
        bar = st.progress(0.0, "Evaluating…")
        grid = ev._grid(room, step)
        grid = [g for g in grid if (room.id == "car" and 0.7 <= g[1] <= 2.2) or (room.id == "cinema" and 6 <= g[1] <= 26.5)]
        cfg = ev.EvalConfig(room_id, layout, tuple(sel), cal, align, level, refl, grid=grid)
        df = ev.evaluate(cfg, hrtf, progress=lambda f: bar.progress(f, f"Evaluating… {100*f:.0f}%"))
        bar.empty()
        st.session_state["eval"] = df
    df = st.session_state.get("eval")
    if df is not None and len(df) and df.attrs.get("room") == room_id:
        st.subheader("Summary")
        summ = ev.summarise(df)
        summ.index = [REGISTRY[i].name if i in REGISTRY else i for i in summ.index]
        st.dataframe(summ, width="stretch")
        st.caption(f"Sweet-spot seats: share of seats whose mean lateral error is at most {ev.SWEET_SPOT_DEG:.0f} deg.")
        metrics = {"lateral_error": ("Lateral localisation error (deg)", 40, "RdYlGn_r"),
                   "arrival_spread_ms": ("Arrival spread (ms)", None, "Magma_r"),
                   "level_vs_ref_db": ("Loudness vs reference seat (dB)", 9, "RdBu_r"),
                   "colouration_db": ("Colouration (dB)", None, "RdYlGn_r"),
                   "envelopment": ("Envelopment, 1 - IACC", 1, "Viridis")}
        m1, m2 = st.columns(2)
        metric = m1.selectbox("Map", list(metrics), format_func=lambda m: metrics[m][0])
        probe = m2.selectbox("Probe sound", ["All probes (mean)"] + list(ev.PROBES[room_id]))
        d = df if probe.startswith("All") else df[df.probe == probe]
        d = d.groupby(["renderer", "x", "y"])[metric].mean().reset_index()
        title, vmax, cs = metrics[metric]
        vmax = vmax or float(np.nanpercentile(d[metric], 95))
        cols = st.columns(min(4, d.renderer.nunique()))
        for k, rn in enumerate(d.renderer.unique()):
            sub = d[d.renderer == rn].pivot(index="y", columns="x", values=metric)
            f = go.Figure(go.Heatmap(z=sub.values, x=sub.columns, y=sub.index, colorscale=cs,
                                     zmin=-vmax if metric == "level_vs_ref_db" else 0, zmax=vmax,
                                     colorbar=dict(title=dict(text=title, side="right"), thickness=10)))
            f.add_trace(go.Scatter(x=[room.ref[0]], y=[room.ref[1]], mode="markers", marker=dict(symbol="cross", size=10, color="black"), showlegend=False))
            f.update_layout(title=dict(text=REGISTRY[rn].name if rn in REGISTRY else rn, font=dict(size=13)), height=430,
                            margin=dict(l=5, r=5, t=40, b=5), xaxis=dict(range=[room.x0, room.x1]),
                            yaxis=dict(range=[room.y1, room.y0], scaleanchor="x"))
            cols[k % len(cols)].plotly_chart(f, width="stretch")
        with tempfile.TemporaryDirectory() as td:
            p = ev.write_report(df, Path(td))
            st.download_button("Download full HTML report", p.read_bytes(), file_name=f"sweetspot_{room_id}_{layout}.html")
        st.download_button("Download results CSV", df.to_csv(index=False).encode(), file_name=f"sweetspot_{room_id}_{layout}.csv")
        with st.expander("How the metrics are computed"):
            st.text(ev.M.__doc__)


# ---------------------------------------------------------------- write tab
with tab_write:
    st.markdown("""
Every algorithm is a small Python class. It gets **where a sound should be** and returns **a gain and an extra delay
for each loudspeaker**. The engine handles calibration, the room, reflections, the subwoofer and the binaural render.

1. Copy `user_renderers/my_renderer.py` to a new file in the same folder and rename the class.
2. Change `drive()`. Keep `@register` above the class.
3. Press **Reload my algorithms** below. It appears in A/B and in the evals.
4. Batch evals from a terminal: `python -m sweetspot.evals.run --room cinema --renderers MyClass VBAP`.
""")
    if st.button("Reload my algorithms"):
        errs = load_user_renderers()
        st.session_state.pop("renders", None)
        if errs:
            st.error("\n\n".join(errs))
        else:
            st.success("Reloaded: " + ", ".join(REGISTRY))
    st.code((ROOT / "user_renderers" / "my_renderer.py").read_text(), language="python")
    st.markdown("Reference implementations to read: `sweetspot/renderers/reference.py` (VBAP, snap, Ambisonics, WFS).")


# ---------------------------------------------------------------- how tab
with tab_how:
    st.markdown(f"""
**Signal path.** Scene sound → your algorithm (gain and delay per speaker, updated every 12 ms for moving sounds)
→ calibration (time alignment, level trim) → each speaker's binaural room impulse response at your seat → headphones.

**Each speaker's response** = direct sound + wall reflections up to order {room.reflection_order} (image-source method)
with distance delay and loss, air absorption, wall absorption and speaker directivity, each arriving from its own
direction through the HRTF ({hrtf.name}), plus a late reverb tail (RT60 {room.rt60:.2f} s). Tweeter and woofer
crossover at 2.5 kHz; the subwoofer gets everything below 100 Hz plus the effects' LFE.

**Assumptions.** Shoebox rooms; typical, guessed dimensions and speaker positions; a generic dummy-head HRTF unless
you load your own SOFA file; no head tracking; directivity is a simple rule, not measured polar data.

**Sounds.** All sounds are generated by the lab, and voices come from the espeak-ng speech synthesiser. Use
"My own audio file" to drop in real recordings.
""")
