"""Run the eval suite: every renderer x every seat on a grid x a set of probe sources.

    python -m sweetspot.evals.run --room cinema --layout 7.1 --out reports/cinema_7.1

writes results.csv (one row per renderer, seat and probe), summary.csv and report.html.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..acoustics import BRIROptions
from ..engine import Seat, SystemSetup
from ..hrtf import HRTF
from ..renderers import REGISTRY, load_user_renderers
from ..rooms import ROOMS, eval_grid
from . import metrics as M

PROBES = {
    "cinema": {
        "Dialogue, screen centre": (0.0, 0.6, 4.0),
        "Between L and C": (-3.0, 0.6, 4.0),
        "Screen left": (-6.0, 0.6, 4.0),
        "Left side wall": (-8.8, 15.0, 5.0),
        "Rear right": (5.0, 27.8, 5.0),
    },
    "car": {
        "Dash centre": (0.0, 0.3, 1.0),
        "Between FL and C": (-0.3, 0.3, 1.0),
        "Front left": (-0.6, 0.3, 1.0),
        "Right side": (0.75, 1.4, 1.0),
        "Rear centre": (0.0, 2.4, 1.0),
    },
}

SWEET_SPOT_DEG = 10.0  # a seat is in the sweet spot if its mean lateral error is at most this


@dataclass
class EvalConfig:
    room: str = "cinema"
    layout: str | None = None
    renderers: tuple = ()
    cal_target: str | None = None
    align: bool = True
    level: bool = True
    reflections: bool = True
    grid_step: float | None = None
    grid: list | None = None        # explicit seat list, overrides grid_step


def _diffuse_points(room):
    c = room.ref3()
    R = 9.0 if room.id == "cinema" else 0.9
    return [c + np.array([R * np.sin(a), -R * np.cos(a), 0.5]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]


def evaluate(cfg: EvalConfig, hrtf: HRTF | None = None, progress=None) -> pd.DataFrame:
    load_user_renderers()
    hrtf = hrtf or HRTF.kemar()
    room = ROOMS[cfg.room]
    layout = cfg.layout or room.default_layout
    names = list(cfg.renderers) or list(REGISTRY)
    renderers = {n: REGISTRY[n]() for n in names}
    setup = SystemSetup(room, layout, cfg.cal_target or next(iter(room.cal_points)), cfg.align, cfg.level,
                        BRIROptions(reflections=cfg.reflections, late_tail=False))
    grid = cfg.grid or (eval_grid(room) if cfg.grid_step is None else _grid(room, cfg.grid_step))
    probes = PROBES[room.id]
    rows = []
    ref_level = {}
    for k, (x, y) in enumerate(grid + [tuple(room.ref)]):
        is_ref = k == len(grid)
        lis = room.listener(x, y)
        seat = Seat(setup, lis, hrtf)
        for rn, r in renderers.items():
            r.prepare(seat.ctx)
            diffuse = [seat.system_ir(r, p) for p in _diffuse_points(room)]
            env = M.envelopment(diffuse)
            for pn, p in probes.items():
                p = np.asarray(p, float)
                h = seat.system_ir(r, p)
                az, el, dist = lis.to_head(p)
                ideal = hrtf.nearest(az, el).astype(float)
                lat_t = M.target_lateral(az, el)
                lat = M.lateral_angle(h, hrtf)
                arr = [a for a in seat.arrivals(r, p) if not a["is_sub"]]
                amax = max(a["amplitude"] for a in arr) or 1
                act = [a["arrival_ms"] for a in arr if a["amplitude"] > 0.1 * amax]
                lev = M.level_db(h)
                if is_ref:
                    ref_level[(rn, pn)] = lev
                    continue
                rows.append(dict(renderer=rn, x=x, y=y, probe=pn, target_lateral=lat_t, perceived_lateral=lat,
                                 lateral_error=abs(lat - lat_t), arrival_spread_ms=max(act) - min(act) if act else 0.0,
                                 level_db=lev, colouration_db=M.colouration_db(h, ideal), envelopment=env))
        if progress:
            progress((k + 1) / (len(grid) + 1))
    df = pd.DataFrame(rows)
    df["level_vs_ref_db"] = df.apply(lambda r: r.level_db - ref_level[(r.renderer, r.probe)], axis=1)
    df.attrs.update(room=room.id, layout=layout, cal_target=setup.cal_target, align=cfg.align, level=cfg.level,
                    reflections=cfg.reflections, hrtf=hrtf.name)
    return df


def _grid(room, step):
    xs = np.arange(room.x0 + step / 2, room.x1, step)
    ys = np.arange(room.y0 + step / 2, room.y1, step)
    return [(float(x), float(y)) for y in ys for x in xs]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    seat = df.groupby(["renderer", "x", "y"]).agg(err=("lateral_error", "mean")).reset_index()
    area = seat.groupby("renderer")["err"].apply(lambda e: 100 * float((e <= SWEET_SPOT_DEG).mean()))
    g = df.groupby("renderer")
    out = pd.DataFrame({
        "Sweet-spot seats (%)": area,
        "Mean lateral error (deg)": g["lateral_error"].mean(),
        "90th pct lateral error (deg)": g["lateral_error"].quantile(0.9),
        "Mean arrival spread (ms)": g["arrival_spread_ms"].mean(),
        "Level spread across seats (dB, std)": g["level_vs_ref_db"].std(),
        "Colouration (dB)": g["colouration_db"].mean(),
        "Envelopment (1-IACC)": g["envelopment"].mean(),
    })
    return out.sort_values("Sweet-spot seats (%)", ascending=False).round(2)


# ---------------------------------------------------------------- report
def _fig_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def heatmaps(df: pd.DataFrame, metric: str, probe: str | None = None, vmax=None, cmap="RdYlGn_r", title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    room = ROOMS[df.attrs.get("room", "cinema")]
    d = df if probe is None else df[df.probe == probe]
    d = d.groupby(["renderer", "x", "y"])[metric].mean().reset_index()
    rs = list(dict.fromkeys(d.renderer))
    fig, axes = plt.subplots(1, len(rs), figsize=(2.6 * len(rs), 3.6 if room.id == "cinema" else 4.2), squeeze=False)
    vmax = vmax or float(np.nanpercentile(d[metric], 95))
    xs = np.sort(d.x.unique())
    ys = np.sort(d.y.unique())
    for ax, rn in zip(axes[0], rs):
        sub = d[d.renderer == rn].pivot(index="y", columns="x", values=metric).reindex(index=ys, columns=xs)
        im = ax.imshow(sub.values, origin="upper", cmap=cmap, vmin=0 if metric != "level_vs_ref_db" else -vmax, vmax=vmax,
                       extent=[xs[0], xs[-1], ys[-1], ys[0]], aspect="equal", interpolation="nearest")
        ax.set_title(REGISTRY[rn].name if rn in REGISTRY else rn, fontsize=8)
        ax.plot([room.ref[0]], [room.ref[1]], "k+", ms=8)
        ax.set_xlim(room.x0, room.x1)
        ax.set_ylim(room.y1, room.y0)
        ax.tick_params(labelsize=6)
        ax.axhline(room.y0, color="k", lw=3)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=title or metric)
    return fig


def write_report(df: pd.DataFrame, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv")
    (out / "config.json").write_text(json.dumps(df.attrs, indent=2, default=str))
    sections = []
    figs = [("Lateral localisation error, mean over probes (deg)", "lateral_error", None, 40, "RdYlGn_r"),
            ("Arrival spread (ms)", "arrival_spread_ms", None, None, "magma_r"),
            ("Loudness relative to the reference seat (dB)", "level_vs_ref_db", None, 9, "coolwarm"),
            ("Colouration (dB)", "colouration_db", None, None, "RdYlGn_r")]
    for probe in PROBES[df.attrs["room"]]:
        figs.append((f"Lateral error for '{probe}' (deg)", "lateral_error", probe, 40, "RdYlGn_r"))
    for title, metric, probe, vmax, cmap in figs:
        fig = heatmaps(df, metric, probe, vmax, cmap, title)
        sections.append(f"<h3>{title}</h3><img src='data:image/png;base64,{_fig_png(fig)}'>")
        plt.close(fig)
    a = df.attrs
    html = f"""<!doctype html><meta charset="utf-8"><title>Sweet-spot evals: {a['room']} {a['layout']}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1b2230}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#eef}}</style>
<h1>Sweet-spot evals: {a['room']}, layout {a['layout']}</h1>
<p>Calibrated for <b>{a['cal_target']}</b> (alignment {a['align']}, level {a['level']}), reflections {a['reflections']},
HRTF {a['hrtf']}. {df[['x','y']].drop_duplicates().shape[0]} seats x {df.probe.nunique()} probe sources.
A seat is in the sweet spot when its mean lateral error is at most {SWEET_SPOT_DEG:.0f} deg. Maps: screen or dashboard at the top, + is the mix reference point.</p>
<h2>Summary</h2>{summ.to_html()}
<h2>Maps</h2>{''.join(sections)}
<h2>How each metric is computed</h2><pre>{M.__doc__}</pre>"""
    p = out / "report.html"
    p.write_text(html)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room", default="cinema", choices=list(ROOMS))
    ap.add_argument("--layout")
    ap.add_argument("--renderers", nargs="*", default=[])
    ap.add_argument("--cal-target")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-level", action="store_true")
    ap.add_argument("--no-reflections", action="store_true")
    ap.add_argument("--grid-step", type=float)
    ap.add_argument("--out", default="reports/latest")
    a = ap.parse_args(argv)
    t = time.time()
    df = evaluate(EvalConfig(a.room, a.layout, tuple(a.renderers), a.cal_target, not a.no_align, not a.no_level,
                             not a.no_reflections, a.grid_step),
                  progress=lambda f: print(f"\r{100*f:5.1f}%", end="", flush=True))
    print(f"\n{len(df)} rows in {time.time()-t:.0f} s")
    print(summarise(df).to_string())
    print("report:", write_report(df, Path(a.out)))


if __name__ == "__main__":
    main()
