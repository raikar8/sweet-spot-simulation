# Sweet Spot Simulation

A listening lab for spatial audio in cinemas and car cabins. Pick a room, a speaker layout, a seat and a scene (characters talking, a forest, a storm with a jet flyover, a truck overtaking in the rain), and hear on **headphones** what that seat really gets. Compare two rendering algorithms A/B, write your own algorithm in Python, and run evals that map where each one works.

There are two versions:

| | Python lab (`app/`, `sweetspot/`) | Browser lab (`index.html`) |
| --- | --- | --- |
| Audio | Offline render: direct sound, wall reflections, reverb, HRTF convolution | Real-time Web Audio, direct sound only |
| Content | Dialogue, nature and cinema-effect scenes, or your own WAV | Test signals |
| Algorithms | Plug-in Python classes, including your own | Fixed set |
| Evals | Per-seat metrics, maps, HTML report | Predicted map only |

## Python lab

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens in your browser with four tabs:

- **Listen and compare**: click a seat on the map, choose a scene and two algorithms, and press Render. Both versions play loudness-matched so only the spatial difference remains. The table shows every speaker's distance, added delay, arrival time, level and off-axis angle at your seat.
- **Sweet-spot evals**: runs every algorithm at every seat on a grid and shows the summary table and maps. Download the HTML report or the CSV.
- **Write your algorithm**: the template and how to plug it in.
- **How it works**: the signal path and assumptions.

Use the sidebar to change the calibration point, time alignment, level trim, reflections, head turn, binaural or plain stereo, and to load your own HRTF (.sofa).

### Write your own rendering algorithm

Copy `user_renderers/my_renderer.py`, rename the class and change `drive()`:

```python
@register
class MyPanner(Renderer):
    name = "My panner"
    def drive(self, source_pos, ctx):          # where the sound should be -> what each speaker plays
        gains = ...                             # one gain per loudspeaker (ctx.speakers)
        delays = np.zeros(ctx.n)                # optional extra delay per loudspeaker, seconds
        return Drive(gains, delays)
```

The engine handles calibration, room acoustics, the subwoofer and the binaural render. Reference implementations to read are in `sweetspot/renderers/reference.py`: VBAP (3-D, triangulated), snap to nearest speaker, first-order Ambisonics, and wave field synthesis.

### Evals

```bash
python -m sweetspot.evals.run --room cinema --layout 7.1 --out reports/cinema_7.1
python -m sweetspot.evals.run --room car --renderers MyPanner VBAP --cal-target "Front left"
pytest
```

Each seat and probe source gets these measures, computed from the binaural impulse response at the ears (details in `sweetspot/evals/metrics.py`):

- **Lateral localisation error**: perceived left-right angle (from ITD and ILD of the first 1.5 ms) against the intended one.
- **Arrival spread**: time from the first to the last speaker arrival.
- **Loudness relative to the reference seat**.
- **Colouration**: how much the spectrum deviates from a real source, which shows comb filtering.
- **Envelopment**: 1 - IACC of a diffuse bed.

Summary: sweet-spot area (share of seats with mean error ≤ 10°), mean and 90th-percentile error. Sample reports are in `reports/`.

### Package layout

```
sweetspot/geometry.py      coordinates, Speaker, Room, Listener
sweetspot/rooms.py         cinema and car definitions, speaker layouts, seats
sweetspot/acoustics.py     image-source reflections, directivity, BRIRs, calibration
sweetspot/hrtf.py          MIT KEMAR HRTF (built in) or your SOFA file
sweetspot/engine.py        renders a scene through a speaker system at one seat
sweetspot/scenes.py        dialogue, nature and effects scenes
sweetspot/synth.py         procedural sounds (rain, wind, birds, jet, thunder, explosion, ...)
sweetspot/renderers/       Renderer interface + reference algorithms
sweetspot/evals/           metrics, batch runner, HTML report
user_renderers/            your algorithms (loaded automatically)
app/streamlit_app.py       the UI
```

### Credits and assumptions

- HRTF: MIT KEMAR dummy-head measurements, Bill Gardner and Keith Martin, MIT Media Lab (1994), used with acknowledgement as its authors ask.
- Voices are synthesised with espeak-ng (`tools/make_assets.py`). All other sounds are generated in code, so the repository holds no third-party recordings.
- The rooms are shoeboxes with typical, guessed dimensions and speaker positions. Speaker directivity uses a simple rule, not measured data. Calibration covers delay and level only, with no EQ.

## Browser lab

Open `index.html` in a browser and press **Start sound**. It needs no install and no dependencies.
